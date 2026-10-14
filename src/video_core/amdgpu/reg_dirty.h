// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>

#include "common/types.h"
#include "video_core/amdgpu/regs.h"

namespace AmdGpu {

/// Track changed blocks of 32 register words so stage B can update its own register file; CLEAR_STATE resets that copy before applying
/// later blocks; used only by the command processor thread
class RegDirty {
public:
    static constexpr u32 BlockWords = 32;
    static constexpr u32 NumBlocks = Regs::NumRegs / BlockWords;
    static_assert(Regs::NumRegs % BlockWords == 0);

    RegDirty() {
        // Send all registers in the first packet to initialize stage B's copy
        for (u32 block = 0; block < NumBlocks; ++block) {
            MarkBlock(block);
        }
    }

    void Mark(u32 first_word, u32 count) noexcept {
        if (count == 0 || first_word >= Regs::NumRegs) {
            return;
        }
        const u32 last = std::min<u32>(first_word + count, Regs::NumRegs) - 1;
        for (u32 block = first_word / BlockWords; block <= last / BlockWords; ++block) {
            MarkBlock(block);
        }
    }

    /// Mark a regs field assigned by a draw packet such as IT_INDEX_TYPE or IT_DRAW_INDEX_2
    template <typename T>
    void MarkField(const Regs& regs, const T& field) noexcept {
        const auto offset = reinterpret_cast<const u8*>(&field) -
                            reinterpret_cast<const u8*>(regs.reg_array.data());
        Mark(static_cast<u32>(offset / sizeof(u32)),
             static_cast<u32>((offset % sizeof(u32) + sizeof(T) + sizeof(u32) - 1) / sizeof(u32)));
    }

    void MarkReset() noexcept {
        reset = true;
    }

    [[nodiscard]] u32 NumDirty() const noexcept {
        return num_dirty;
    }
    [[nodiscard]] bool Reset() const noexcept {
        return reset;
    }
    [[nodiscard]] const u16* Blocks() const noexcept {
        return blocks.data();
    }

    /// Start a fresh block list after the packet takes these changes
    void Clear() noexcept {
        for (u32 i = 0; i < num_dirty; ++i) {
            const u16 block = blocks[i];
            bits[block / 64] &= ~(u64{1} << (block % 64));
        }
        num_dirty = 0;
        reset = false;
    }

private:
    void MarkBlock(u32 block) noexcept {
        u64& word = bits[block / 64];
        const u64 bit = u64{1} << (block % 64);
        if ((word & bit) == 0) {
            word |= bit;
            blocks[num_dirty++] = static_cast<u16>(block);
        }
    }

    std::array<u64, (NumBlocks + 63) / 64> bits{};
    std::array<u16, NumBlocks> blocks{};
    u32 num_dirty{};
    bool reset{};
};

} // namespace AmdGpu
