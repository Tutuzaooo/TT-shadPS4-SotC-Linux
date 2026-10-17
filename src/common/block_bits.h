// SPDX-FileCopyrightText: Copyright 2026 TT-shadPS4 SotC Linux
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace Common {

/// Fixed-size bit set with fast set-bit iteration. Replaces std::bitset where the code walks the
/// set bits with _Find_first/_Find_next, which only libstdc++ provides (libc++ builds need this).
template <size_t N>
class BlockBits {
    static constexpr size_t Words = (N + 63) / 64;
    std::array<uint64_t, Words> words{};

public:
    static constexpr size_t size() noexcept {
        return N;
    }
    void set(size_t i) noexcept {
        words[i / 64] |= uint64_t{1} << (i % 64);
    }
    void reset() noexcept {
        words.fill(0);
    }
    bool test(size_t i) const noexcept {
        return (words[i / 64] >> (i % 64)) & 1;
    }
    bool any() const noexcept {
        for (const uint64_t w : words) {
            if (w) {
                return true;
            }
        }
        return false;
    }
    bool none() const noexcept {
        return !any();
    }
    size_t count() const noexcept {
        size_t n = 0;
        for (const uint64_t w : words) {
            n += static_cast<size_t>(std::popcount(w));
        }
        return n;
    }
    /// First set bit, or size() if none
    size_t _Find_first() const noexcept {
        for (size_t w = 0; w < Words; ++w) {
            if (words[w]) {
                return w * 64 + static_cast<size_t>(std::countr_zero(words[w]));
            }
        }
        return N;
    }
    /// Next set bit after prev, or size() if none
    size_t _Find_next(size_t prev) const noexcept {
        ++prev;
        if (prev >= N) {
            return N;
        }
        size_t w = prev / 64;
        uint64_t bits = words[w] & (~uint64_t{0} << (prev % 64));
        while (true) {
            if (bits) {
                return w * 64 + static_cast<size_t>(std::countr_zero(bits));
            }
            if (++w >= Words) {
                return N;
            }
            bits = words[w];
        }
    }
    BlockBits& operator|=(const BlockBits& other) noexcept {
        for (size_t w = 0; w < Words; ++w) {
            words[w] |= other.words[w];
        }
        return *this;
    }
    friend BlockBits operator|(BlockBits a, const BlockBits& b) noexcept {
        a |= b;
        return a;
    }
    bool operator==(const BlockBits&) const = default;
};

} // namespace Common
