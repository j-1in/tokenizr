#ifndef BPE_BPE_H_
#define BPE_BPE_H_

#include <cstddef>
#include <string>
#include <vector>

namespace bpe {

using Byte = unsigned char;

struct WordCount {
    std::vector<Byte> word;
    std::size_t count = 0;
};

struct CharSplit {
    std::vector<Byte> chars;
    std::size_t count = 0;
};

struct TokenCount {
    std::vector<Byte> token;
    std::size_t count = 0;
};

struct Results {
    std::vector<WordCount> word_counts;
    std::vector<CharSplit> char_splits;
    std::vector<TokenCount> tokens;
};

struct Word {
    const Byte* bytes;
};

void parallel_task1(std::vector<Byte>& input, Results& results);

void parallel_task2(const std::vector<CharSplit>& splits, Results& results);

std::vector<Byte> read_file(const std::string& path);

std::vector<Word> split_words(std::vector<Byte>& input);

std::vector<Word> parallel_split_words(std::vector<Byte>& input);

void parallel_scan(std::vector<int>& starts, std::vector<int>& word_id);

void task1(const std::vector<Word>& words, Results& results);

void task2(const std::vector<CharSplit>& splits, Results& results);

void write_output(const Results& results, const std::string& path);

void print_task1(const Results& results);

Results run_pipeline(std::vector<Byte>& input);

int run_cli(int argc, char** argv);

}

#endif
