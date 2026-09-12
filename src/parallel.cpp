#include "bpe.h"
#include "word_key.h"
#include "word_counts.h"
#include "absl/log/log.h"
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <array>
#include <limits>
#include <string_view>
#include <utility>

namespace bpe {

namespace {

constexpr std::size_t shard_count = 256;

using detail::WordKey;
using Counts = detail::WordCounts;
using ShardSummary = std::array<std::size_t, shard_count>;
using CountEntry = std::pair<WordKey, std::size_t>;

struct ThreadBlock {
    std::size_t tid;
    std::size_t nthreads;
    std::size_t beg;
    std::size_t end;
};

struct RoutedCounts {
    std::vector<CountEntry> entries;
    std::array<std::size_t, shard_count + 1> offsets{};
    std::size_t largest_shard = 0;
};

inline ThreadBlock current_thread_block(std::size_t size) {
    const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
    const std::size_t nthreads = static_cast<std::size_t>(omp_get_num_threads());
    const std::size_t base = size / nthreads;
    const std::size_t rem = size % nthreads;
    const std::size_t beg = tid * base + std::min(tid, rem);
    const std::size_t end = beg + base + (tid < rem ? 1 : 0);
    return ThreadBlock{tid, nthreads, beg, end};
}

inline bool is_separator(Byte b) { return b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D; }

std::int64_t elapsed_ms(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::vector<Counts> local_count_words(const std::vector<Byte>& input, std::size_t& occurrences) {
    std::vector<Counts> local_counts;
    occurrences = 0;  // Total occurrences across all threads

    #pragma omp parallel shared(local_counts) reduction(+ : occurrences)
    {
        const ThreadBlock block = current_thread_block(input.size());

        #pragma omp single
        local_counts.resize(block.nthreads);

        auto& counts = local_counts[block.tid];

        std::size_t i = block.beg;

        // Find next separator or reach end of chunk if starting in middle of a word.
        if (i > 0 && i < block.end && !is_separator(input[i - 1])) {
            while (i < block.end && !is_separator(input[i])) ++i;
        }

        // Count words in thread's chunk.
        while (i < block.end) {
            while (i < block.end && is_separator(input[i])) ++i;  // Skip consecutive separators
            if (i == block.end) break;
            const std::size_t word_beg = i;
            while (i < input.size() && !is_separator(input[i])) ++i;  // Find end of word
            ++counts[WordKey{std::string_view(
                reinterpret_cast<const char*>(input.data() + word_beg), i - word_beg)}];
            ++occurrences;
        }
    }

    return local_counts;
}

RoutedCounts route_counts(std::vector<Counts>& local_counts, std::size_t local_distinct) {
    const auto shard_of = [](const WordKey& word) {
        return word.hash >> (std::numeric_limits<std::size_t>::digits - 8);
    };

    std::vector<ShardSummary> histogram(local_counts.size());

    #pragma omp parallel for schedule(static)
    for (std::size_t t = 0; t < local_counts.size(); ++t) {
        for (const auto& entry : local_counts[t]) {
            ++histogram[t][shard_of(entry.first)];
        }
    }

    RoutedCounts routed;
    routed.entries.resize(local_distinct);

    std::vector<ShardSummary> cursors(local_counts.size());

    for (std::size_t shard = 0; shard < shard_count; ++shard) {
        std::size_t offset = routed.offsets[shard];

        for (std::size_t t = 0; t < local_counts.size(); ++t) {
            cursors[t][shard] = offset;
            offset += histogram[t][shard];
        }

        routed.offsets[shard + 1] = offset;
        routed.largest_shard = std::max(routed.largest_shard, offset - routed.offsets[shard]);
    }

    #pragma omp parallel for schedule(static)
    for (std::size_t t = 0; t < local_counts.size(); ++t) {
        for (const auto& entry : local_counts[t]) {
            routed.entries[cursors[t][shard_of(entry.first)]++] = entry;
        }
        Counts{}.swap(local_counts[t]);
    }

    return routed;
}

std::vector<Counts> reduce_shards(const RoutedCounts& routed) {
    std::vector<Counts> shards(shard_count);

    #pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t shard = 0; shard < shard_count; ++shard) {
        auto& counts = shards[shard];

        for (std::size_t i = routed.offsets[shard]; i < routed.offsets[shard + 1]; ++i) {
            counts[routed.entries[i].first] += routed.entries[i].second;
        }
    }  // Each complete shard has one owner; no shared map growth occurs.

    return shards;
}
}

void parallel_task1(std::vector<Byte>& input, Results& results) {
    const auto count_start = std::chrono::steady_clock::now();

    std::size_t occurrences = 0;
    auto local_counts = local_count_words(input, occurrences);

    const auto local_end = std::chrono::steady_clock::now();

    std::size_t local_distinct = 0;
    for (const auto& counts : local_counts) {
        local_distinct += counts.size();
    }

    // Route equal byte strings to the same shard.
    // Prefix only P*S summaries. Each local map gets its own interval within
    // each shard, so scatter requires neither atomics nor concurrent push_back.
    auto routed = route_counts(local_counts, local_distinct);

    const auto route_end = std::chrono::steady_clock::now();

    auto shards = reduce_shards(routed);
    std::vector<CountEntry>().swap(routed.entries);
    std::array<std::size_t, shard_count + 1> output_offsets{};

    for (std::size_t shard = 0; shard < shard_count; ++shard) {
        output_offsets[shard + 1] = output_offsets[shard] + shards[shard].size();
    }

    const auto count_end = std::chrono::steady_clock::now();

    const std::size_t distinct = output_offsets.back();

    // Allocate and populate results
    results.word_counts.resize(distinct);
    results.char_splits.resize(distinct);

    #pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t shard = 0; shard < shard_count; ++shard) {
        std::size_t i = output_offsets[shard];
        for (const auto& entry : shards[shard]) {
            const auto* bytes = reinterpret_cast<const Byte*>(entry.first.bytes.data());
            results.word_counts[i].word.assign(bytes, bytes + entry.first.bytes.size());
            results.word_counts[i].count = entry.second;
            results.char_splits[i].chars.assign(bytes, bytes + entry.first.bytes.size());
            results.char_splits[i].count = entry.second;
            ++i;
        }
        // Output construction includes releasing the corresponding shard map.
        Counts{}.swap(shards[shard]);
    }

    const auto copy_end = std::chrono::steady_clock::now();
    LOG(INFO) << "word count: " << elapsed_ms(count_start, count_end) << " ms";
    LOG(INFO) << "char split: " << elapsed_ms(count_end, copy_end) << " ms";
    LOG(INFO) << "fused discovery and local aggregation: " << elapsed_ms(count_start, local_end)
              << " ms; shard routing: " << elapsed_ms(local_end, route_end)
              << " ms; shard reduction: " << elapsed_ms(route_end, count_end) << " ms";
    LOG(INFO) << "reduction shards: " << shard_count
              << "; largest shard entries: " << routed.largest_shard;
    LOG(INFO) << "aggregation occurrences M: " << occurrences
              << "; local distinct D: " << local_distinct << "; global distinct U: " << distinct;
}

[[deprecated]] std::vector<Word> parallel_split_words(std::vector<Byte>& input) {
    const std::size_t input_size = input.size();
    input.push_back(Byte('\0'));
    std::vector<std::size_t> counts;
    std::vector<std::size_t> offsets;
    std::vector<Word> words;

    #pragma omp parallel shared(counts, offsets, words)
    {
        const ThreadBlock block = current_thread_block(input_size);

        #pragma omp single
        {
            counts.resize(block.nthreads);
            offsets.resize(block.nthreads);
        }

        std::size_t local_count = 0;

        // Save the predecessor before any thread rewrites separators.
        const bool initial_prev_is_sep =
            (block.beg == 0) ? true : is_separator(input[block.beg - 1]);

        bool prev_is_sep = initial_prev_is_sep;
        for (std::size_t i = block.beg; i < block.end; i++) {
            const bool curr_is_sep = is_separator(input[i]);
            if (prev_is_sep && !curr_is_sep) {
                local_count++;
            }
            prev_is_sep = curr_is_sep;
        }
        counts[block.tid] = local_count;

        // All counts and boundary reads must finish before prefixing/writing.
        #pragma omp barrier

        #pragma omp single
        {
            std::size_t total = 0;
            for (std::size_t t = 0; t < counts.size(); ++t) {
                offsets[t] = total;
                total += counts[t];
            }
            words.resize(total);
        }

        // Read each owned byte before normalizing it, never reread a neighbour
        std::size_t out = offsets[block.tid];
        prev_is_sep = initial_prev_is_sep;
        for (std::size_t i = block.beg; i < block.end; ++i) {
            const bool curr_is_sep = is_separator(input[i]);
            if (prev_is_sep && !curr_is_sep) {
                words[out++] = Word{input.data() + i};
            }
            if (curr_is_sep) {
                input[i] = Byte('\0');
            }
            prev_is_sep = curr_is_sep;
        }
    }

    return words;
}

void parallel_task2(const std::vector<CharSplit>& splits, Results& results) {
    task2(splits, results);
}

}
