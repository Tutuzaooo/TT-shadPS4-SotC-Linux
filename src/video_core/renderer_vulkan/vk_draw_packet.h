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

/// Transfer stage A's draw or dispatch state to stage B for binding and recording; stage B reads registers, extents and user data from this
/// packet; 8-byte aligned sections contain Header, u16 block indices, register words, each Stage with its user data and flat data, then a
/// dispatch's ComputeProgram
namespace DrawPacket {

enum class Kind : u8 {
    Skip,     ///< No pipeline; stage B only processes pending operations
    Filtered, ///< FilterDraw replaces this draw with a clear, resolve or copy
    Draw,
    DrawIndirect,
    Dispatch,
    DispatchIndirect,
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
    u32 index_offset;
    const Pipeline* pipeline;
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> cb_extent;
    AmdGpu::CbDbExtent db_extent;
    /// DrawIndirect stores adjusted arguments and an optional count; DispatchIndirect stores stride bytes of arguments
    VAddr address;
    VAddr count_address;
    u32 stride;
    u32 max_count;
};

struct Stage {
    const Shader::Info* info;
    VAddr pgm_base;
    u32 ud_size;
    u32 flat_size;
};

constexpr u32 Align(u32 size) {
    return (size + 7) & ~7u;
}

} // namespace DrawPacket

} // namespace Vulkan
