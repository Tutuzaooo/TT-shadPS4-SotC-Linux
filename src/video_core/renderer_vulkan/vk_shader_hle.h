// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <span>
#include <vector>

#include "common/types.h"

namespace AmdGpu {
struct ComputeProgram;
union Regs;
} // namespace AmdGpu

namespace Shader {
struct Info;
}

namespace Vulkan {

class Rasterizer;

/// SotC's replicate_fill_hle shader fills small tables with dst[i] = src[i % m]
constexpr u64 ReplicateFillHash = 0x50784801;

/// Keep fills above this limit on the GPU
constexpr u64 ReplicateFillMaxBytes = 64_KB;

enum class ReplicateFillPlan : u8 {
    Fill,    ///< Write data to dst
    Empty,   ///< No elements need writing
    Refused, ///< Leave this fill on the GPU
};

/// Describe the fill's control, source and destination addresses, source and destination byte sizes, n and m, user SGPR 6 and dispatched
/// thread count
struct ReplicateFillShape {
    VAddr ctl;
    VAddr src;
    VAddr dst;
    u64 src_size;
    u64 dst_size;
    u32 n;
    u32 m;
    u32 limit;
    u64 threads;
};

/// Track the destination element count and source element count
struct ReplicateFillSize {
    ReplicateFillPlan plan;
    u64 count;
    u64 src_count;
};

/// Compute fill sizes from the dispatch shape; reject large fills and shapes that could differ from GPU behavior
constexpr ReplicateFillSize SizeReplicateFill(const ReplicateFillShape& shape) {
    constexpr ReplicateFillSize Refused{ReplicateFillPlan::Refused, 0, 0};
    // Require aligned dwords because GPU arena bindings discard the low address bits
    if (((shape.ctl | shape.src | shape.dst) & 3) != 0) {
        return Refused;
    }
    // Write dst[i] = src[i % m] up to min(n, user SGPR 6, dispatched threads); discard stores beyond the destination
    const u64 count = std::min<u64>({shape.n, shape.limit, shape.threads, shape.dst_size / 4});
    if (count == 0) {
        return {ReplicateFillPlan::Empty, 0, 0};
    }
    // Keep large fills on the GPU; SotC's small walker tables matched, but a 4 MiB loading fill failed verification
    if (count > ReplicateFillMaxBytes / 4) {
        return Refused;
    }
    // Leave division by zero to the GPU, where it reads src[i]
    if (shape.m == 0) {
        return Refused;
    }
    // GPU reads beyond the source return zero
    const u64 src_count = std::min<u64>(shape.m, count);
    if (src_count * 4 > shape.src_size) {
        return Refused;
    }
    // Reject overlap because GPU threads may read source or control data while other threads overwrite it
    const auto over_dst = [&shape, count](VAddr addr, u64 size) {
        return addr < shape.dst + count * 4 && shape.dst < addr + size;
    };
    if (over_dst(shape.src, src_count * 4) || over_dst(shape.ctl, 8)) {
        return Refused;
    }
    return {ReplicateFillPlan::Fill, count, src_count};
}

/// Repeat the first src_count elements through data using doubling copies rather than per-element division
inline void RepeatReplicateFill(std::span<u32> data, u64 src_count) {
    if (src_count == 0) {
        return;
    }
    for (u64 done = src_count; done < data.size();) {
        const u64 chunk = std::min<u64>(done, data.size() - done);
        std::memcpy(data.data() + done, data.data(), chunk * sizeof(u32));
        done += chunk;
    }
}

/// Build fill output from guest memory and the packet's user data, setting dst after reading its V#; reject unsupported shapes, metadata
/// destinations and control or source data that is GPU-written or not GPU-mapped
ReplicateFillPlan PlanReplicateFill(const Shader::Info& info,
                                    const AmdGpu::ComputeProgram& cs_program,
                                    Rasterizer& rasterizer, VAddr& dst, std::vector<u32>& data);

/// Attempts to execute a shader using HLE if possible.
bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::Regs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer);

} // namespace Vulkan
