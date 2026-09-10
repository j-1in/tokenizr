#include "bpe.h"
#include "absl/log/log.h"
#include <chrono>
#include <omp.h>
#include <algorithm>

namespace bpe {

namespace {
inline bool is_separator(Byte b) {
    return b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D || b == 0x00;
}

}

namespace {
std::int64_t elapsed_ms(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}
}

void parallel_task1(std::vector<Byte>& input, Results& results) {
    const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();

    const std::vector<Word> words = parallel_split_words(input);

    const std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    LOG(INFO) << "split words: " << elapsed_ms(t0, t1) << " ms";
    task1(words, results);
}

std::vector<Word> parallel_split_words(std::vector<Byte>& input) {
    input.push_back(Byte('\0'));
    const std::size_t n = input.size();
    std::vector<std::size_t> counts;
    std::vector<std::size_t> offsets;
    std::vector<Word> words;

    #pragma omp parallel shared(counts, offsets, words)
    {
        const int tid = omp_get_thread_num();
        const int p = omp_get_num_threads();

        #pragma omp single
        {
            counts.resize(static_cast<std::size_t>(p));
            offsets.resize(static_cast<std::size_t>(p));
        }

        const std::size_t p_size = static_cast<std::size_t>(p);
        const std::size_t tid_size = static_cast<std::size_t>(tid);

        const std::size_t base = n / p_size;
        const std::size_t rem = n % p_size;

        const std::size_t beg = tid_size * base + std::min(tid_size, rem);
        const std::size_t len = base + (tid_size < rem ? 1 : 0);

        const std::size_t end = beg + len;
        std::size_t local_count = 0;

        // Save the predecessor before any thread rewrites separators.
        const bool initial_prev_is_sep =
            (beg == 0) ? true : is_separator(input[beg - 1]);

        bool prev_is_sep = initial_prev_is_sep;
        for (std::size_t i = beg; i < end; i++) {
            const bool curr_is_sep = is_separator(input[i]);
            if (prev_is_sep && !curr_is_sep) {
                local_count++;
            }
            prev_is_sep = curr_is_sep;
        }
        counts[tid_size] = local_count;

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
        std::size_t out = offsets[tid_size];
        prev_is_sep = initial_prev_is_sep;
        for (std::size_t i = beg; i < end; ++i) {
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
