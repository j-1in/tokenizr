#ifndef BPE_WORD_COUNTS_H_
#define BPE_WORD_COUNTS_H_

#include "word_key.h"
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bpe::detail {

// Custom flat hash map implementation using open addressing and linear probing.
// Uses WordKey as key type and insertion only so empty slots indicate unused slots.
class WordCounts {
   public:
    using Entry = std::pair<WordKey, std::size_t>;

    std::size_t size() const noexcept { return size_; }

    std::size_t& operator[](const WordKey& key) {
        if (key.bytes.empty()) throw std::invalid_argument("empty word key");
        if (slots_.empty()) grow();

        auto index = find_slot(key);
        if (!slots_[index].first.bytes.empty()) return slots_[index].second;

        if (size_ >= slots_.size() - slots_.size() / 4) {
            grow();
            index = find_slot(key);
        }

        slots_[index].first = key;
        ++size_;
        return slots_[index].second;
    }

    void swap(WordCounts& other) noexcept {
        slots_.swap(other.slots_);
        std::swap(size_, other.size_);
    }

    class Iterator {
       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Entry;
        using difference_type = std::ptrdiff_t;
        using pointer = const Entry*;
        using reference = const Entry&;

        Iterator() = default;

        reference operator*() const { return *current_; }
        pointer operator->() const { return current_; }

        Iterator& operator++() {
            ++current_;
            skip_empty();
            return *this;
        }

        Iterator operator++(int) {
            auto old = *this;
            ++*this;
            return old;
        }

        bool operator==(const Iterator& other) const { return current_ == other.current_; }
        bool operator!=(const Iterator& other) const { return !(*this == other); }

       private:
        friend class WordCounts;

        Iterator(pointer current, pointer end) : current_(current), end_(end) { skip_empty(); }

        void skip_empty() {
            while (current_ != end_ && current_->first.bytes.empty()) ++current_;
        }

        pointer current_ = nullptr;
        pointer end_ = nullptr;
    };

    Iterator begin() const {
        if (slots_.empty()) return {};
        return {slots_.data(), slots_.data() + slots_.size()};
    }

    Iterator end() const {
        if (slots_.empty()) return {};
        return {slots_.data() + slots_.size(), slots_.data() + slots_.size()};
    }

   private:
    std::size_t find_slot(const WordKey& key) const {
        const std::size_t mask = slots_.size() - 1;
        std::size_t index = key.hash & mask;
        while (!slots_[index].first.bytes.empty() && !WordKeyEqual{}(slots_[index].first, key)) {
            index = (index + 1) & mask;
        }
        return index;
    }

    void grow() {
        const auto old_capacity = slots_.size();

        if (old_capacity > slots_.max_size() / 2)
            throw std::length_error("word table capacity overflow");

        std::vector<Entry> replacement(old_capacity ? old_capacity * 2 : 16);

        slots_.swap(replacement);

        for (const auto& entry : replacement) {
            if (!entry.first.bytes.empty()) slots_[find_slot(entry.first)] = entry;
        }
    }

    std::vector<Entry> slots_;
    std::size_t size_ = 0;
};

}

#endif
