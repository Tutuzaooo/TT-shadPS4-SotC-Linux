// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>

#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"

namespace Shader {
struct Info;
}

namespace Vulkan {

class Pipeline;

/// Transfer stage A's draw or dispatch state to stage B for binding and recording; 8-byte aligned sections contain Header, block indices,
/// register words, each Stage with user data, flat data and optional RingBindings, a dispatch's ComputeProgram, and the full register file
/// when verify is enabled
namespace DrawPacket {

enum class Kind : u8 {
    Skip,     ///< No pipeline; stage B only processes pending operations
    Filtered, ///< FilterDraw replaces this draw with a clear, resolve or copy
    Draw,
    DrawIndirect,
    Dispatch,
    DispatchIndirect,
    Task, ///< Run this draw_pipe_steps task in order through Rasterizer::RunInOrder
};

/// FilterDraw's result for stage B to execute
enum class Filter : u8 {
    None,
    EliminateFastClear,
    FmaskDecompress,
    Resolve,
    PrimitiveTypeNone,
    DepthStencilCopy,
};

struct Header {
    u32 size;
    Kind kind;
    Filter filter;
    u8 is_indexed;
    u8 compute;    ///< A dispatch supplies unordered_ring
    u8 regs_reset; ///< Apply SetDefaults before the blocks after CLEAR_STATE
    u8 num_stages;
    u16 num_blocks;
    s8 unordered_ring;         ///< Call NoteRingCommand for gcn_unordered_dispatches
    u8 game_dispatch_accesses; ///< A guest dispatch without DMA for gcn_unordered_dispatches
    u8 depth_copy;             ///< DepthStencilCopy filter result
    u8 stencil_copy;
    u8 fill_hle; ///< Stage A selected this dispatch as a CPU fill candidate
    u8 verify;   ///< SHADPS4_PIPE_VERIFY includes the full register file for stage B checks
    /// Carry synchronized rings from gcn_unordered_dispatches while the recording thread had pending packets
    u16 game_sync_rings;
    u32 index_offset;
    const Pipeline* pipeline;
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> cb_extent;
    AmdGpu::CbDbExtent db_extent;
    /// DrawIndirect stores adjusted arguments and an optional count; DispatchIndirect stores stride bytes of arguments
    VAddr address;
    VAddr count_address;
    u32 stride;
    u32 max_count;
    /// End offset of built packets' constant-ring allocations; stage B stamps it with the recording submission's tick
    u64 ring_end;
};

/// A task packet starts with size and kind at Header's offsets, followed by a trivially copyable callable copied as bytes
struct TaskHeader {
    u32 size;
    Kind kind;
    void (*run)(const void* task);
};

struct Stage {
    const Shader::Info* info;
    VAddr pgm_base;
    u32 ud_size;
    u32 flat_size;
    u32 num_ring; ///< constant_ring RingBindings follow the data words
};

/// A stage's buffer copied into constant_ring by stage A, in descriptor order
struct RingBinding {
    u32 desc_index; ///< Index into Shader::Info::buffers
    u32 size;
    u64 offset; ///< Byte offset in the ring
    VAddr address;
};

constexpr u32 Align(u32 size) {
    return (size + 7) & ~7u;
}

} // namespace DrawPacket

} // namespace Vulkan
