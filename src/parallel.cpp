#include "bpe.h"
#include "absl/log/log.h"
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <array>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace bpe {

namespace {
inline bool is_separator(Byte b) {
    return b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D || b == 0x00;
}

struct ThreadBlock {
    std::size_t tid;
    std::size_t nthreads;
    std::size_t beg;
    std::size_t end;
};

inline ThreadBlock current_thread_block(std::size_t size) {
    const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
    const std::size_t nthreads =
        static_cast<std::size_t>(omp_get_num_threads());
    const std::size_t base = size / nthreads;
    const std::size_t rem = size % nthreads;
    const std::size_t beg = tid * base + std::min(tid, rem);
    const std::size_t end = beg + base + (tid < rem ? 1 : 0);
    return ThreadBlock{tid, nthreads, beg, end};
}

std::int64_t elapsed_ms(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}

constexpr std::size_t shard_count = 256;
using Counts = std::unordered_map<std::string_view, std::size_t>;
using Summary = std::array<std::size_t, shard_count>;
using Entry = std::pair<std::string_view, std::size_t>;
}

void parallel_task1(std::vector<Byte>& input, Results& results) {
    const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();

    const std::vector<Word> words = parallel_split_words(input);

    const std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    LOG(INFO) << "split words: " << elapsed_ms(t0, t1) << " ms";

    using Counts = std::unordered_map<std::string_view, std::size_t>;
    std::vector<Counts> local_counts;
    const auto count_start = std::chrono::steady_clock::now();

    #pragma omp parallel shared(local_counts)
    {
        const ThreadBlock block = current_thread_block(words.size());

        #pragma omp single
        local_counts.resize(block.nthreads);

        auto& counts = local_counts[block.tid];
        for (std::size_t i = block.beg; i < block.end; ++i) {
            ++counts[std::string_view(
                reinterpret_cast<const char*>(words[i].bytes))];
        }
    }

    const auto local_end = std::chrono::steady_clock::now();
    std::size_t local_distinct = 0;
    for (const auto& counts : local_counts) {
        local_distinct += counts.size();
    }

    // Route equal byte strings to the same shard.
    constexpr std::size_t shard_count = 256;
    const auto shard_of = [](std::string_view word) {
        return std::hash<std::string_view>{}(word) >>
               (std::numeric_limits<std::size_t>::digits - 8);
    };

    using Summary = std::array<std::size_t, shard_count>;
    using Entry = std::pair<std::string_view, std::size_t>;
    std::vector<Summary> histogram(local_counts.size());

    #pragma omp parallel for schedule(static)
    for (std::size_t t = 0; t < local_counts.size(); ++t) {
        for (const auto& entry : local_counts[t]) {
            ++histogram[t][shard_of(entry.first)];
        }
    }

    // Prefix only P*S summaries. Each local map gets its own interval within
    // each shard, so scatter requires neither atomics nor concurrent push_back.
    std::array<std::size_t, shard_count + 1> shard_offsets{};
    std::vector<Summary> cursors(local_counts.size());
    std::size_t largest_shard = 0;

    for (std::size_t shard = 0; shard < shard_count; ++shard) {
        std::size_t offset = shard_offsets[shard];
        for (std::size_t t = 0; t < local_counts.size(); ++t) {
            cursors[t][shard] = offset;
            offset += histogram[t][shard];
        }
        shard_offsets[shard + 1] = offset;
        largest_shard = std::max(largest_shard, offset - shard_offsets[shard]);
    }

    std::vector<Entry> routed(local_distinct);

    #pragma omp parallel for schedule(static)
    for (std::size_t t = 0; t < local_counts.size(); ++t) {
        for (const auto& entry : local_counts[t]) {
            routed[cursors[t][shard_of(entry.first)]++] = entry;
        }
        Counts{}.swap(local_counts[t]);
    }

    const auto route_end = std::chrono::steady_clock::now();
    std::vector<Counts> shards(shard_count);

    #pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t shard = 0; shard < shard_count; ++shard) {
        auto& counts = shards[shard];
        for (std::size_t i = shard_offsets[shard]; i < shard_offsets[shard + 1];
             ++i) {
            counts[routed[i].first] += routed[i].second;
        }
    }  // Each complete shard has one owner; no shared map growth occurs.

    std::vector<Entry>().swap(routed);
    std::array<std::size_t, shard_count + 1> output_offsets{};
    for (std::size_t shard = 0; shard < shard_count; ++shard) {
        output_offsets[shard + 1] =
            output_offsets[shard] + shards[shard].size();
    }

    const auto count_end = std::chrono::steady_clock::now();
    const std::size_t distinct = output_offsets.back();
    results.word_counts.resize(distinct);
    results.char_splits.resize(distinct);

    #pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t shard = 0; shard < shard_count; ++shard) {
        std::size_t i = output_offsets[shard];
        for (const auto& entry : shards[shard]) {
            const auto* bytes =
                reinterpret_cast<const Byte*>(entry.first.data());
            results.word_counts[i].word.assign(bytes,
                                               bytes + entry.first.size());
            results.word_counts[i].count = entry.second;
            results.char_splits[i].chars.assign(bytes,
                                                bytes + entry.first.size());
            results.char_splits[i].count = entry.second;
            ++i;
        }
        // Output construction includes releasing the corresponding shard map.
        Counts{}.swap(shards[shard]);
    }

    const auto copy_end = std::chrono::steady_clock::now();
    LOG(INFO) << "word count: " << elapsed_ms(count_start, count_end) << " ms";
    LOG(INFO) << "char split: " << elapsed_ms(count_end, copy_end) << " ms";
    LOG(INFO) << "local aggregation: " << elapsed_ms(count_start, local_end)
              << " ms; shard routing: " << elapsed_ms(local_end, route_end)
              << " ms; shard reduction: " << elapsed_ms(route_end, count_end)
              << " ms";
    LOG(INFO) << "reduction shards: " << shard_count
              << "; largest shard entries: " << largest_shard;
    LOG(INFO) << "aggregation occurrences M: " << words.size()
              << "; local distinct D: " << local_distinct
              << "; global distinct U: " << distinct;
}

std::vector<Word> parallel_split_words(std::vector<Byte>& input) {
    input.push_back(Byte('\0'));
    std::vector<std::size_t> counts;
    std::vector<std::size_t> offsets;
    std::vector<Word> words;

    #pragma omp parallel shared(counts, offsets, words)
    {
        const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t nthreads =
            static_cast<std::size_t>(omp_get_num_threads());

        const ThreadBlock block = current_thread_block(input.size());

        #pragma omp single
        {
            counts.resize(nthreads);
            offsets.resize(nthreads);
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
        counts[tid] = local_count;

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
        std::size_t out = offsets[tid];
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
