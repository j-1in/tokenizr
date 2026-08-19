#include "bpe.h"
#include "absl/log/log.h"
#include <chrono>

namespace bpe {

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
    const std::vector<Word> words = split_words(input);
    const std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    LOG(INFO) << "split words: " << elapsed_ms(t0, t1) << " ms";
    task1(words, results);
}

void parallel_task2(const std::vector<CharSplit>& splits, Results& results) {
    task2(splits, results);
}

}
