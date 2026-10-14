// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include "core/memory.h"
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"

extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Vulkan {

static constexpr u64 COPY_SHADER_HASH = 0xfefebf9f;

static bool ExecuteCopyShaderHLE(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program,
                                 Rasterizer& rasterizer) {
    auto& runtime = rasterizer.GetRuntime();
    auto& buffer_cache = rasterizer.GetBufferCache();

    // Copy shader defines three formatted buffers as inputs: control, source, and destination.
    const auto ctl_buf_sharp = info.buffers[0].GetSharp(info);
    const auto src_buf_sharp = info.buffers[1].GetSharp(info);
    const auto dst_buf_sharp = info.buffers[2].GetSharp(info);
    const auto buf_stride = src_buf_sharp.GetStride();
    ASSERT(buf_stride == dst_buf_sharp.GetStride());

    struct CopyShaderControl {
        u32 dst_idx;
        u32 src_idx;
        u32 end;
    };
    static_assert(sizeof(CopyShaderControl) == 12);
    ASSERT(ctl_buf_sharp.GetStride() == sizeof(CopyShaderControl));
    const auto ctl_buf = reinterpret_cast<const CopyShaderControl*>(ctl_buf_sharp.base_address);

    static std::vector<vk::BufferCopy> copies;
    copies.clear();
    copies.reserve(cs_program.dim_x);

    for (u32 i = 0; i < cs_program.dim_x; i++) {
        const auto& [dst_idx, src_idx, end] = ctl_buf[i];
        const u32 local_dst_offset = dst_idx * buf_stride;
        const u32 local_src_offset = src_idx * buf_stride;
        const u32 local_size = (end + 1) * buf_stride;
        copies.emplace_back(local_src_offset, local_dst_offset, local_size);
    }

    static constexpr vk::DeviceSize MaxDistanceForMerge = 64_MB;
    u32 batch_start = 0;
    u32 batch_end = 0;

    while (batch_end < copies.size()) {
        // Place first copy into the current batch
        const auto& copy = copies[batch_start];
        auto src_offset_min = copy.srcOffset;
        auto src_offset_max = copy.srcOffset + copy.size;
        auto dst_offset_min = copy.dstOffset;
        auto dst_offset_max = copy.dstOffset + copy.size;

        for (++batch_end; batch_end < copies.size(); batch_end++) {
            // Compute new src and dst bounds if we were to batch this copy
            const auto& [src_offset, dst_offset, size] = copies[batch_end];
            auto new_src_offset_min = std::min(src_offset_min, src_offset);
            auto new_src_offset_max = std::max(src_offset_max, src_offset + size);
            if (new_src_offset_max - new_src_offset_min > MaxDistanceForMerge) {
                break;
            }

            auto new_dst_offset_min = std::min(dst_offset_min, dst_offset);
            auto new_dst_offset_max = std::max(dst_offset_max, dst_offset + size);
            if (new_dst_offset_max - new_dst_offset_min > MaxDistanceForMerge) {
                break;
            }

            // We can batch this copy
            src_offset_min = new_src_offset_min;
            src_offset_max = new_src_offset_max;
            dst_offset_min = new_dst_offset_min;
            dst_offset_max = new_dst_offset_max;
        }

        // Obtain buffers for the total source and destination ranges.
        const auto [src_buf, src_buf_offset] = buffer_cache.ObtainBuffer(
            src_buf_sharp.base_address + src_offset_min, src_offset_max - src_offset_min, false);
        const auto [dst_buf, dst_buf_offset] = buffer_cache.ObtainBuffer(
            dst_buf_sharp.base_address + dst_offset_min, dst_offset_max - dst_offset_min, true);

        // Apply found buffer base.
        const auto vk_copies = std::span{copies}.subspan(batch_start, batch_end - batch_start);
        for (auto& copy : vk_copies) {
            copy.srcOffset = copy.srcOffset - src_offset_min + src_buf_offset;
            copy.dstOffset = copy.dstOffset - dst_offset_min + dst_buf_offset;
        }

        // Execute buffer copies.
        LOG_TRACE(Render_Vulkan, "HLE buffer copy: src_size = {}, dst_size = {}",
                  src_offset_max - src_offset_min, dst_offset_max - dst_offset_min);
        runtime.CopyBuffer(src_buf, dst_buf, vk_copies);
        buffer_cache.CloseGpuWrites();
        batch_start = batch_end;
    }

    return true;
}

ReplicateFillPlan PlanReplicateFill(const Shader::Info& info,
                                    const AmdGpu::ComputeProgram& cs_program,
                                    Rasterizer& rasterizer, VAddr& dst, std::vector<u32>& data) {
    // Based on AjncSystem's ShadowOfTheColossus-PC ExecuteReplicateShaderHLE; uses control (n, m), source and destination buffers with
    // 32-bit elements
    constexpr auto Refused = ReplicateFillPlan::Refused;
    std::array<const Shader::BufferResource*, 3> buffers{};
    u32 num_buffers = 0;
    for (const auto& desc : info.buffers) {
        if (desc.IsSpecial()) {
            continue;
        }
        if (num_buffers == buffers.size()) {
            return Refused;
        }
        buffers[num_buffers++] = &desc;
    }
    if (num_buffers != 3 || buffers[0]->is_written || buffers[1]->is_written ||
        !buffers[2]->is_written) {
        return Refused;
    }
    // Use one row of 64-thread groups; any zero dimension means no work
    if (cs_program.dim_y > 1 || cs_program.dim_z > 1 || cs_program.num_thread_x.full != 64 ||
        cs_program.num_thread_y.full != 1 || cs_program.num_thread_z.full != 1 ||
        cs_program.start_x != 0) {
        return Refused;
    }
    const Shader::UdView ud{info};
    const auto ctl = buffers[0]->GetSharp(ud);
    const auto src = buffers[1]->GetSharp(ud);
    const auto out = buffers[2]->GetSharp(ud);
    // Match idxen addressing: stride 0 makes all threads use element 0, and swizzled V#s reorder elements
    if (src.GetStride() != 4 || out.GetStride() != 4 || src.swizzle_enable || out.swizzle_enable ||
        ctl.GetSize() < 8 || ctl.base_address <= 1 || src.base_address <= 1 ||
        out.base_address <= 1) {
        return Refused;
    }
    dst = out.base_address;
    // IsComputeMetaClear treats a metadata destination as a clear, including fills with no elements
    if (rasterizer.GetTextureCache().IsMeta(dst)) {
        return Refused;
    }
    auto& buffer_cache = rasterizer.GetBufferCache();
    auto* memory = Core::Memory::Instance();
    // BindBuffers binds buffers outside GPU mappings as null, making GPU reads return zero
    if (!rasterizer.IsMapped(ctl.base_address, 8) ||
        buffer_cache.IsRegionGpuModified(ctl.base_address, 8)) {
        return Refused;
    }
    std::array<u32, 2> control{};
    if (!memory->TryReadBacking(ctl.base_address, control.data(), sizeof(control))) {
        return Refused;
    }
    const auto size = SizeReplicateFill({
        .ctl = ctl.base_address,
        .src = src.base_address,
        .dst = dst,
        .src_size = src.GetSize(),
        .dst_size = out.GetSize(),
        .n = control[0],
        .m = control[1],
        .limit = cs_program.user_data[6],
        .threads = u64{cs_program.dim_x} * cs_program.dim_y * cs_program.dim_z * 64,
    });
    if (size.plan != ReplicateFillPlan::Fill) {
        return size.plan;
    }
    const u64 src_bytes = size.src_count * sizeof(u32);
    if (!rasterizer.IsMapped(src.base_address, src_bytes) ||
        buffer_cache.IsRegionGpuModified(src.base_address, src_bytes)) {
        return Refused;
    }
    data.resize(size.count);
    if (!memory->TryReadBacking(src.base_address, data.data(), src_bytes)) {
        return Refused;
    }
    RepeatReplicateFill(data, size.src_count);
    return ReplicateFillPlan::Fill;
}

bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::Regs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer) {
    switch (info.pgm_hash) {
    case COPY_SHADER_HASH:
        return ExecuteCopyShaderHLE(info, cs_program, rasterizer);
    default:
        return false;
    }
}

} // namespace Vulkan
