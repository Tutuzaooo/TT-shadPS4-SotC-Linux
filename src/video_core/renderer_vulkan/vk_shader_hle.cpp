// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/texture_cache.h"

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

// Shadow of the Colossus fills small tables every frame with a compute shader:
// dst[i] = src[i % m] (32-bit elements) for i < min(n, user SGPR 6, the dispatched threads),
// n and m read from the constant buffer. The command processor then reads those tables (SRT
// walker), and with the shader on the GPU every such read waits for the GPU (readback). With a
// CPU-written source the fill is done here, in command order, so the tables stay CPU-side.
// After the SotC PC port (vk_shader_hle.cpp, SOTC_FILL_HLE): about a third of its readbacks.
// PINK_FILL_HLE=0 runs the shader on the GPU.
static constexpr u64 REPLICATE_SHADER_HASH = 0x50784801;

namespace FillStats {
std::atomic<u64> done{}, bytes{}, gpu{}, big{}, big_bytes{}, ns{}, mirrored{}, mirror_bytes{},
    mirror_ns{};
std::array<std::atomic<u64>, 8> fallback{};
// 0 shape, 1 control GPU-written, 2 source GPU-written, 3 foreign GPU bytes on the pages,
// 4 stack pages, 5 not backed, 6 CPU writes waiting for an upload, 7 over PINK_FILL_HLE_MAX_KB
void Report() {
    static auto last = std::chrono::steady_clock::now();
    static u32 calls = 0;
    if ((++calls & 63) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - last).count();
    if (seconds < 10.0) {
        return;
    }
    last = now;
    u64 f[8];
    for (u32 i = 0; i < 8; ++i) {
        f[i] = fallback[i].exchange(0, std::memory_order_relaxed);
    }
    LOG_WARNING(Render_Vulkan,
                "Fill HLE: {:.0f}/s on the CPU ({:.0f} KB/s, {:.1f} ms/s; over 64 KiB: {:.0f}/s, "
                "{:.0f} KB/s), {:.0f}/s on the GPU (copied to memory too: {:.0f}/s, {:.0f} KB/s, "
                "{:.1f} ms/s); fallbacks/s: shape {:.0f}, control {:.0f}, "
                "source {:.0f}, pages {:.0f}, stack {:.0f}, backing {:.0f}, upload {:.0f}, "
                "size cap {:.0f}",
                done.exchange(0) / seconds, bytes.exchange(0) / 1024.0 / seconds,
                ns.exchange(0) / 1e6 / seconds, big.exchange(0) / seconds,
                big_bytes.exchange(0) / 1024.0 / seconds, gpu.exchange(0) / seconds,
                mirrored.exchange(0) / seconds, mirror_bytes.exchange(0) / 1024.0 / seconds,
                mirror_ns.exchange(0) / 1e6 / seconds,
                f[0] / seconds, f[1] / seconds, f[2] / seconds, f[3] / seconds, f[4] / seconds,
                f[5] / seconds, f[6] / seconds, f[7] / seconds);
}
} // namespace FillStats

// A large fill left to the GPU: its values also go to guest memory once the dispatch is
// recorded, so reads of the table on the CPU need no readback (FinishShaderHLE)
struct PendingMirror {
    VAddr address{};
    u64 count{};
    const u32* source{};
    u64 pattern{};
};
static PendingMirror pending_mirror{};

static bool ExecuteReplicateShaderHLE(const Shader::Info& info,
                                      const AmdGpu::ComputeProgram& cs_program,
                                      Rasterizer& rasterizer) {
    static const bool enabled = [] {
        const char* v = std::getenv("PINK_FILL_HLE");
        const bool on = !v || v[0] != '0';
        LOG_WARNING(Render_Vulkan, "Fill HLE (cs {:#x} on the CPU): {}", REPLICATE_SHADER_HASH,
                    on ? "on" : "off");
        return on;
    }();
    FillStats::Report();
    const auto fallback = [](u32 reason) {
        FillStats::fallback[reason].fetch_add(1, std::memory_order_relaxed);
        FillStats::gpu.fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    if (!enabled) {
        FillStats::gpu.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (info.buffers.size() < 3 || cs_program.dim_y > 1 || cs_program.dim_z > 1) {
        return fallback(0);
    }
    auto& buffer_cache = rasterizer.GetBufferCache();
    const auto ctl = info.buffers[0].GetSharp(info);
    const auto src = info.buffers[1].GetSharp(info);
    const auto dst = info.buffers[2].GetSharp(info);
    const u32 src_stride = src.GetStride() ? src.GetStride() : 4;
    const u32 dst_stride = dst.GetStride() ? dst.GetStride() : 4;
    if (src_stride != 4 || dst_stride != 4 || ctl.GetSize() < 8 || ctl.base_address <= 1 ||
        src.base_address <= 1 || dst.base_address <= 1) {
        return fallback(0);
    }
    if (buffer_cache.IsRegionGpuModified(ctl.base_address, 8)) {
        return fallback(1);
    }
    const u32* control = std::bit_cast<const u32*>(ctl.base_address);
    const u32 n = control[0];
    const u32 m = control[1];
    const u64 threads = u64(cs_program.dim_x) * 64;
    const u64 count = std::min<u64>({n, cs_program.user_data[6], threads, dst.GetSize() / 4});
    if (count == 0) {
        FillStats::done.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    const u64 src_count = m ? std::min<u64>(m, count) : count;
    if (src_count * 4 > src.GetSize()) {
        return fallback(0);
    }
    // PINK_FILL_HLE_MAX_KB=<n>: larger fills stay on the GPU (default 64; 0: no limit)
    static const u64 max_bytes = [] {
        const char* v = std::getenv("PINK_FILL_HLE_MAX_KB");
        return (v ? std::strtoull(v, nullptr, 10) : 64) * 1024;
    }();
    if (buffer_cache.IsRegionGpuModified(src.base_address, src_count * 4)) {
        return fallback(2);
    }
    if (max_bytes != 0 && count * 4 > max_bytes) {
        // PINK_FILL_MIRROR=1: the large fills also go to guest memory (off by default)
        static const bool mirror = [] {
            const char* v = std::getenv("PINK_FILL_MIRROR");
            return v && v[0] == '1';
        }();
        if (mirror) {
            pending_mirror = {dst.base_address, count, std::bit_cast<const u32*>(src.base_address),
                              src_count};
        }
        return fallback(7);
    }
    const auto start = std::chrono::steady_clock::now();
    const u32* source = std::bit_cast<const u32*>(src.base_address);
    const VAddr address = dst.base_address;
    if (const u32 reason = buffer_cache.TryFillFromCpu(address, count, source, src_count);
        reason != 0) {
        return fallback(reason + 2);
    }
    rasterizer.GetTextureCache().InvalidateMemory(address, count * 4);
    static u32 logged = 0;
    if (logged < 6) {
        ++logged;
        LOG_WARNING(Render_Vulkan, "Fill HLE: {:#x}+{:#x} from {:#x} ({} source elements)",
                    address, count * 4, src.base_address, src_count);
    }
    FillStats::done.fetch_add(1, std::memory_order_relaxed);
    FillStats::bytes.fetch_add(count * 4, std::memory_order_relaxed);
    if (count * 4 > 64 * 1024) {
        FillStats::big.fetch_add(1, std::memory_order_relaxed);
        FillStats::big_bytes.fetch_add(count * 4, std::memory_order_relaxed);
    }
    FillStats::ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count(),
                            std::memory_order_relaxed);
    return true;
}

void FinishShaderHLE(Rasterizer& rasterizer) {
    if (pending_mirror.count == 0) {
        return;
    }
    const PendingMirror mirror = std::exchange(pending_mirror, {});
    const auto start = std::chrono::steady_clock::now();
    if (rasterizer.GetBufferCache().MirrorFillToCpu(mirror.address, mirror.count, mirror.source,
                                                    mirror.pattern)) {
        FillStats::mirrored.fetch_add(1, std::memory_order_relaxed);
        FillStats::mirror_bytes.fetch_add(mirror.count * 4, std::memory_order_relaxed);
        FillStats::mirror_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - start)
                                           .count(),
                                       std::memory_order_relaxed);
    }
}

bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::Regs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer) {
    pending_mirror = {};
    switch (info.pgm_hash) {
    case COPY_SHADER_HASH:
        return ExecuteCopyShaderHLE(info, cs_program, rasterizer);
    case REPLICATE_SHADER_HASH:
        return ExecuteReplicateShaderHLE(info, cs_program, rasterizer);
    default:
        return false;
    }
}

} // namespace Vulkan
