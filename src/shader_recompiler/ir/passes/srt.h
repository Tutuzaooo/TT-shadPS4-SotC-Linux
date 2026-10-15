// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

struct SrtWalkerContext;
/// rdx and rcx are unused; r8 carries the context the generated guest loads check (SrtWalkerContext)
using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/,
                                            u64 /*unused*/, u64 /*unused*/,
                                            const SrtWalkerContext* /*context*/);
/// Copies a walker into the executable buffer, sharing identical code. Returns nullptr when the buffer is
/// full
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);
/// Bytes used in the walker code buffer and bytes saved by sharing identical walkers
size_t GetSrtCodeUsage();
size_t GetSrtCodeShared();

/// A faulting SRT load that can be completed from guest memory
struct SrtLoad {
    u64 address; // Memory operand address
    u32 size;    // Pointer or dword size
    u32 length;  // Instruction size
};
std::optional<SrtLoad> DecodeSrtLoad(void* context);
/// Writes `value` to the load's destination register and skips the instruction
void CompleteSrtLoad(void* context, const SrtLoad& load, u64 value);

/// Fast path for clean walker reads from pages seen by the fault handler
/// The generated code reads these fields at fixed offsets
struct SrtWalkerContext {
    u64* clean_pages;                                                 // offset 0
    bool PS4_SYSV_ABI (*clean_load)(u64 address, u32 size, u64* out); // offset 8
    void (*copy_window)(u32* dst, u64 src, u64 bytes);                // offset 16
};

/// ReadConst flag (without direct memory access): a load whose dword index is only known while
/// the shader runs (a loop counter). The walker copies a window of the table into the flattened
/// buffer; the low bits hold the window's first dword, and the shader clamps the index to it.
/// After the SotC fork of AjncSystem (609b4dd2): SotC's depth of field reads its sample offsets
/// this way, and without it draws black blocks over nearby objects
constexpr u32 ReadConstDynamicWindowFlag = 1u << 31;
constexpr u32 ReadConstDynamicWindowDwords = 512;
/// Whether walkers build those windows: direct memory access off and PINK_DMA_WINDOW not 0
bool ReadConstWindows();
constexpr u64 SrtCleanPageLimit = 1ULL << 40;
const SrtWalkerContext* GetSrtWalkerContext();
/// Sets the clean-load callback owned by the page manager
void SetSrtCleanLoad(bool PS4_SYSV_ABI (*clean_load)(u64, u32, u64*));
/// Marks pages that can try the clean-load fast path
void MarkSrtCleanPages(u64 address, u32 size);

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
