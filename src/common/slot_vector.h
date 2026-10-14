// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <numeric>
#include <utility>
#include <vector>
#include "common/assert.h"
#include "common/types.h"

namespace Common {

struct SlotId {
    static constexpr u32 INVALID_INDEX = std::numeric_limits<u32>::max();

    SlotId() noexcept = default;
    constexpr SlotId(u32 index) noexcept : index(index) {}

    constexpr auto operator<=>(const SlotId&) const noexcept = default;

    constexpr explicit operator bool() const noexcept {
        return index != INVALID_INDEX;
    }

    u32 index = INVALID_INDEX;
};

/// Keep entries in fixed chunks so inserts and erases never move other entries; growth fills a fixed chunk table, and other threads can
/// read its stable pointers once an inserted id has been published to them
template <class T>
class SlotVector {
    constexpr static std::size_t ChunkBits = 10;
    constexpr static std::size_t ChunkSize = std::size_t{1} << ChunkBits;
    constexpr static std::size_t ChunkMask = ChunkSize - 1;
    constexpr static std::size_t MaxChunks = 1024;

public:
    template <typename ValueType, typename Pointer, typename Reference>
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = ValueType;
        using difference_type = std::ptrdiff_t;
        using pointer = Pointer;
        using reference = Reference;

        Iterator(SlotVector& vector_, SlotId index_) : vector(vector_), slot(index_) {
            AdvanceToValid();
        }

        reference operator*() const {
            return vector[slot];
        }

        pointer operator->() const {
            return &vector[slot];
        }

        Iterator& operator++() {
            ++slot.index;
            AdvanceToValid();
            return *this;
        }

        Iterator operator++(int) {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const Iterator& other) const {
            return slot == other.slot;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        void AdvanceToValid() {
            while (slot < vector.values_capacity && !vector.ReadStorageBit(slot.index)) {
                ++slot.index;
            }
        }

        SlotVector& vector;
        SlotId slot;
    };

    using iterator = Iterator<T, T*, T&>;
    using const_iterator = Iterator<const T, const T*, const T&>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    SlotVector() = default;

    SlotVector(const SlotVector&) = delete;
    SlotVector& operator=(const SlotVector&) = delete;

    ~SlotVector() noexcept {
        for (std::size_t c = 0; c < values_capacity >> ChunkBits; ++c) {
            Chunk* const chunk = chunks[c];
            std::size_t index = 0;
            for (u64 bits : chunk->stored_bitset) {
                for (std::size_t bit = 0; bits; ++bit, bits >>= 1) {
                    if ((bits & 1) != 0) {
                        chunk->values[index + bit].object.~T();
                    }
                }
                index += 64;
            }
            delete chunk;
        }
    }

    [[nodiscard]] T& operator[](SlotId id) noexcept {
        ValidateIndex(id);
        return Value(id.index).object;
    }

    [[nodiscard]] const T& operator[](SlotId id) const noexcept {
        ValidateIndex(id);
        return Value(id.index).object;
    }

    bool is_allocated(SlotId id) const {
        const std::size_t c = id.index >> ChunkBits;
        return c < MaxChunks && chunks[c] && ReadStorageBit(id.index);
    }

    template <typename... Args>
    SlotId insert(Args&&... args) noexcept {
        // Growing the chunk table keeps existing entries and references in args valid
        const u32 index = FreeValueIndex();
        new (&Value(index).object) T(std::forward<Args>(args)...);
        SetStorageBit(index);

        return SlotId{index};
    }

    void erase(SlotId id) noexcept {
        Value(id.index).object.~T();
        free_list.push_back(id.index);
        ResetStorageBit(id.index);
    }

    std::size_t size() const noexcept {
        return values_capacity - free_list.size();
    }

    iterator begin() noexcept {
        return iterator(*this, 0);
    }

    const_iterator begin() const noexcept {
        return const_iterator(*this, 0);
    }

    const_iterator cbegin() const noexcept {
        return begin();
    }

    iterator end() noexcept {
        return iterator(*this, values_capacity);
    }

    const_iterator end() const noexcept {
        return const_iterator(*this, values_capacity);
    }

    const_iterator cend() const noexcept {
        return end();
    }

    reverse_iterator rbegin() noexcept {
        return reverse_iterator(end());
    }

    const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(end());
    }

    const_reverse_iterator crbegin() const noexcept {
        return rbegin();
    }

    reverse_iterator rend() noexcept {
        return reverse_iterator(begin());
    }

    const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(begin());
    }

    const_reverse_iterator crend() const noexcept {
        return rend();
    }

private:
    struct NonTrivialDummy {
        NonTrivialDummy() noexcept {}
    };

    union Entry {
        Entry() noexcept : dummy{} {}
        ~Entry() noexcept {}

        NonTrivialDummy dummy;
        T object;
    };

    struct Chunk {
        Chunk() noexcept : stored_bitset{} {}

        Entry values[ChunkSize];
        std::array<u64, ChunkSize / 64> stored_bitset;
    };

    [[nodiscard]] Entry& Value(u32 index) noexcept {
        return chunks[index >> ChunkBits]->values[index & ChunkMask];
    }

    [[nodiscard]] const Entry& Value(u32 index) const noexcept {
        return chunks[index >> ChunkBits]->values[index & ChunkMask];
    }

    [[nodiscard]] u64& StorageWord(u32 index) noexcept {
        return chunks[index >> ChunkBits]->stored_bitset[(index & ChunkMask) / 64];
    }

    [[nodiscard]] const u64& StorageWord(u32 index) const noexcept {
        return chunks[index >> ChunkBits]->stored_bitset[(index & ChunkMask) / 64];
    }

    void SetStorageBit(u32 index) noexcept {
        StorageWord(index) |= u64(1) << (index % 64);
    }

    void ResetStorageBit(u32 index) noexcept {
        StorageWord(index) &= ~(u64(1) << (index % 64));
    }

    bool ReadStorageBit(u32 index) const noexcept {
        return ((StorageWord(index) >> (index % 64)) & 1) != 0;
    }

    void ValidateIndex([[maybe_unused]] SlotId id) const noexcept {
        DEBUG_ASSERT(id);
        DEBUG_ASSERT(id.index < values_capacity);
        DEBUG_ASSERT(ReadStorageBit(id.index));
    }

    [[nodiscard]] u32 FreeValueIndex() noexcept {
        if (free_list.empty()) {
            AddChunk();
        }

        const u32 free_index = free_list.back();
        free_list.pop_back();
        return free_index;
    }

    void AddChunk() noexcept {
        const std::size_t c = values_capacity >> ChunkBits;
        if (c == MaxChunks) [[unlikely]] {
            UNREACHABLE_MSG("SlotVector is full with {} entries", values_capacity);
        }
        chunks[c] = new Chunk;

        // Return the lowest new index first
        const std::size_t old_free_size = free_list.size();
        free_list.resize(old_free_size + ChunkSize);
        std::iota(free_list.rbegin(), free_list.rbegin() + ChunkSize,
                  static_cast<u32>(values_capacity));

        values_capacity += ChunkSize;
    }

    std::array<Chunk*, MaxChunks> chunks{};
    std::size_t values_capacity = 0;

    std::vector<u32> free_list;
};

} // namespace Common

template <>
struct std::hash<Common::SlotId> {
    std::size_t operator()(const Common::SlotId& id) const noexcept {
        return std::hash<u32>{}(id.index);
    }
};
