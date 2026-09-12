#ifndef BPE_WORD_KEY_H_
#define BPE_WORD_KEY_H_

#include <cstddef>
#include <functional>
#include <string_view>

namespace bpe::detail {

// Replaces std::string_view in map keys to avoid repeated hashing of the same string_view.
struct WordKey {
    std::string_view bytes;
    std::size_t hash = 0;

    WordKey() = default;  // No hashing for empty key
    explicit WordKey(std::string_view word)
        : bytes(word), hash(std::hash<std::string_view>{}(word)) {}
};

struct WordKeyHash {
    std::size_t operator()(const WordKey& word) const noexcept { return word.hash; }
};

// Check both the hash and the bytes for equality to avoid collisions.
struct WordKeyEqual {
    bool operator()(const WordKey& a, const WordKey& b) const noexcept {
        return a.hash == b.hash && a.bytes == b.bytes;
    }
};

}

#endif
