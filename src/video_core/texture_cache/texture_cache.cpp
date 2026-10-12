// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <optional>
// XXH3_state_t for the streaming hash of HashGuestMemory
#define XXH_STATIC_LINKING_ONLY
#include <xxhash.h>

#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/guest_write_journal.h"
#include "common/hash.h"
#include "common/logging/events.h"
#include "common/perf_stats.h"
#include "common/scope_exit.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_draw_pipe.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr u64 PageShift = 12;
static constexpr u64 NumFramesBeforeRemoval = 32;
// GC ticks an image must stay unused before the GC frees it
// below the pressure threshold
[[maybe_unused]] static constexpr u64 GcIdleTicks = 16;
[[maybe_unused]] static constexpr u64 GcFullScanIdleTicks = 300; // texture_gc_full_scan

/// Hashes guest memory through its physical backing, like XXH3_64bits over the guest range. A direct read can
/// fault, and inside the fault handler a nested fault on memory the tracker does not handle (not GPU mapped,
/// or with no backing) was a host access violation (SotC, §17 of the port guide). Returns nullopt when part
/// of the range has no physical backing, or when the memory manager lock is held for writing (possibly by the
/// faulting thread itself)
static std::optional<u64> HashGuestMemory(VAddr addr, u64 size) {
    auto* memory = Core::Memory::Instance();
    XXH3_state_t state;
    XXH3_64bits_reset(&state);
    while (size != 0) {
        VAddr run_start{};
        u64 run_size{};
        const u8* run = memory->GetBackingRun(addr, &run_start, &run_size, true);
        if (!run) {
            return std::nullopt;
        }
        const u64 offset = addr - run_start;
        const u64 chunk = std::min(size, run_size - offset);
        XXH3_64bits_update(&state, run + offset, chunk);
        addr += chunk;
        size -= chunk;
    }
    return XXH3_64bits_digest(&state);
}

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                           BufferCache& buffer_cache_, PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_}, blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, runtime, buffer_cache.GetStreamBuffer()},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()},
      gc_full_scan{EmulatorSettings.IsTextureGcFullScan()} {

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
    } else {
        s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
        // PINK_VRAM_LIMIT_MB=<n>: collect images as if the card had only n MiB to spend.
        if (const char* v = std::getenv("PINK_VRAM_LIMIT_MB")) {
            const s64 limit = std::strtoll(v, nullptr, 10) * s64{1_MB};
            if (limit >= s64{1_GB}) {
                device_local_memory = std::min(device_local_memory, limit);
            }
        }
        const s64 min_spacing_expected = device_local_memory - 1_GB;
        const s64 min_spacing_critical = device_local_memory - 512_MB;
        const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
        const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
        const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
        pressure_gc_memory = static_cast<u64>(std::max<u64>(
            std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
            DEFAULT_PRESSURE_GC_MEMORY));
        critical_gc_memory = static_cast<u64>(std::max<u64>(
            std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
            DEFAULT_CRITICAL_GC_MEMORY));
        trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
        // Buffer arena memory competes for the same budget, so on cards with 8 GB or less start
        // aggressive image collection well before the budget is reached.
        small_vram = device_local_memory <= s64{8_GB};
        if (small_vram) {
            critical_gc_memory = std::min<u64>(critical_gc_memory, device_local_memory * 6 / 10);
            pressure_gc_memory = std::min<u64>(pressure_gc_memory, critical_gc_memory);
        }
        LOG_WARNING(Render, "Image GC: budget {} MiB, pressure {} MiB, critical {} MiB",
                    device_local_memory >> 20, pressure_gc_memory >> 20, critical_gc_memory >> 20);
    }

    if (const u32 trigger_mb = EmulatorSettings.GetTextureGcTriggerMb(); trigger_mb != 0) {
        trigger_gc_memory = u64{trigger_mb} * 1_MB;
        LOG_WARNING(Render_Vulkan, "Workaround texture_gc_trigger_mb enabled: {} MB", trigger_mb);
    }
    if (gc_full_scan) {
        LOG_WARNING(Render_Vulkan, "Workaround texture_gc_full_scan enabled");
    }
    LOG_INFO(Render_Vulkan, "Texture GC thresholds: trigger {} MB, pressure {} MB, critical {} MB",
             trigger_gc_memory / 1_MB, pressure_gc_memory / 1_MB, critical_gc_memory / 1_MB);
}

TextureCache::~TextureCache() = default;

void TextureCache::ProcessDownloadImages() {
    Vulkan::DrawPipe::CheckOwner("TextureCache::ProcessDownloadImages");
    std::unique_lock lk{download_images_mutex};
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, true);
    }
    download_images.clear();
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    const u32 download_size = image.info.pitch * image.info.size.height * image.info.size.depth *
                              image.info.resources.layers * (image.info.num_bits / 8);
    ASSERT(download_size <= image.info.guest_size);
    Common::PerfStats::Add(Common::PerfStats::Id::ImageDownloads);
    const auto download =
        runtime.GetStagingPool().Request(download_size, MemoryType::HostCached, 16, !sync);
    const vk::BufferImageCopy image_download = {
        .bufferOffset = download.offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&image_download, 1});
    if (sync) {
        scheduler.Finish();
        download.Invalidate();
        Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::ImageDownload,
                                          image.info.guest_address, download_size,
                                          download.mapped, 1);
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image.info.guest_address),
                                                  download.mapped, download_size);
    } else {
        scheduler.DeferPriorityOperation(
            [this, device_addr = image.info.guest_address, download, download_size] {
                download.Invalidate();
                Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::ImageDownload,
                                                  device_addr, download_size, download.mapped, 0);
                Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(device_addr),
                                                          download.mapped, download_size);
                runtime.GetStagingPool().FreeDeferred(download);
            });
    }
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash. This runs inside the guest's write fault handler, after the buffer cache brought
        // the page up to date, so the backing holds the current contents
        const auto hash = HashGuestMemory(image.info.guest_address, image.info.guest_size);
        if (!hash) {
            // No backing to compare against later: treat the image as modified
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
            return;
        }
        image.hash = *hash;
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    std::scoped_lock lock{mutex};
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    StampWrite(pages_start, pages_end - pages_start);
    ForEachImageInRegion(pages_start, pages_end - pages_start, [&](ImageId image_id, Image& image) {
        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    });
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    std::scoped_lock lock{mutex};
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            return;
        }
        // Ensure image is reuploaded when accessed again.
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::scoped_lock lk{mutex};
    StampWrite(cpu_addr, size);

    ImageIds deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        // Each extent separately: std::max over SubresourceExtent is lexicographic, it would keep the cached
        // layer count when the request has fewer levels but more layers
        new_info.resources.levels =
            std::max(requested_info.resources.levels, cache_image.info.resources.levels);
        new_info.resources.layers =
            std::max(requested_info.resources.layers, cache_image.info.resources.layers);
        new_info.UpdateSize();
        const auto new_image_id = slot_images.insert(instance, runtime, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;
        runtime.CopyColorAndDepth(&cache_image, &new_image);

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NumFramesBeforeRemoval;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        const bool pow2_padding_only =
            image_info.props.is_pow2 != cache_image.info.props.is_pow2 &&
            image_info.tile_mode == cache_image.info.tile_mode &&
            image_info.size == cache_image.info.size &&
            image_info.pitch == cache_image.info.pitch && image_info.resources.levels == 1 &&
            cache_image.info.resources.levels == 1 && image_info.resources.layers == 1 &&
            cache_image.info.resources.layers == 1;

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size || pow2_padding_only) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_RENDER_PROBLEM(
            Render_Vulkan, Error,
            "Unresolvable image overlap with equal memory address:\n"
            "=== OLD IMAGE (cached) ===\n"
            "  Address:        {:#x}\n"
            "  Size:           {:#x} bytes\n"
            "  Format:         {}\n"
            "  Type:           {}\n"
            "  Width:          {}\n"
            "  Height:         {}\n"
            "  Depth:          {}\n"
            "  Pitch:          {}\n"
            "  Mip levels:     {}\n"
            "  Array layers:   {}\n"
            "  Samples:        {}\n"
            "  Tile mode:      {:#x}\n"
            "  Block size:     {} bits\n"
            "  Is block-comp:  {}\n"
            "  Guest size:     {:#x}\n"
            "  Last accessed:  tick {}\n"
            "  Safe to delete: {}\n"
            "  isPow2:         {}\n"
            "  Alt tile:       {}\n"
            "\n"
            "=== NEW IMAGE (requested) ===\n"
            "  Address:        {:#x}\n"
            "  Size:           {:#x} bytes\n"
            "  Format:         {}\n"
            "  Type:           {}\n"
            "  Width:          {}\n"
            "  Height:         {}\n"
            "  Depth:          {}\n"
            "  Pitch:          {}\n"
            "  Mip levels:     {}\n"
            "  Array layers:   {}\n"
            "  Samples:        {}\n"
            "  Tile mode:      {:#x}\n"
            "  Block size:     {} bits\n"
            "  Is block-comp:  {}\n"
            "  Guest size:     {:#x}\n"
            "  isPow2:         {}\n"
            "  Alt tile:       {}\n"
            "\n"
            "=== COMPARISON ===\n"
            "  Same format:           {}\n"
            "  Same type:             {}\n"
            "  Same tile mode:        {}\n"
            "  Same block size:       {}\n"
            "  Same BlockDim:         {}\n"
            "  Same pitch:            {}\n"
            "  Same pow2:             {}\n"
            "  Same alt tile:         {}\n"
            "  Old resources <= new:  {} (old: {}, new: {})\n"
            "  Old size <= new size:  {}\n"
            "  Expected size (calc):  {} bytes\n"
            "  Size ratio (new/expected): {:.2f}x\n"
            "  Size ratio (new/old):  {:.2f}x\n"
            "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
            "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
            "  Merged image ID:       {}\n"
            "  Binding type:          {}\n"
            "  Current tick:          {}\n"
            "  Age (ticks since last access): {}",

            // Old image details
            cache_image.info.guest_address, cache_image.info.guest_size,
            vk::to_string(cache_image.info.pixel_format), static_cast<int>(cache_image.info.type),
            cache_image.info.size.width, cache_image.info.size.height, cache_image.info.size.depth,
            cache_image.info.pitch, cache_image.info.resources.levels,
            cache_image.info.resources.layers, cache_image.info.num_samples,
            static_cast<u32>(cache_image.info.tile_mode), cache_image.info.num_bits,
            +cache_image.info.props.is_block, cache_image.info.guest_size,
            cache_image.tick_accessed_last, safe_to_delete, bool(cache_image.info.props.is_pow2),
            cache_image.info.alt_tile,

            // New image details
            image_info.guest_address, image_info.guest_size, vk::to_string(image_info.pixel_format),
            static_cast<int>(image_info.type), image_info.size.width, image_info.size.height,
            image_info.size.depth, image_info.pitch, image_info.resources.levels,
            image_info.resources.layers, image_info.num_samples,
            static_cast<u32>(image_info.tile_mode), image_info.num_bits, image_info.props.is_block,
            image_info.guest_size, bool(image_info.props.is_pow2), image_info.alt_tile,

            // Comparison
            (image_info.pixel_format == cache_image.info.pixel_format),
            (image_info.type == cache_image.info.type),
            (image_info.tile_mode == cache_image.info.tile_mode),
            (image_info.num_bits == cache_image.info.num_bits),
            (image_info.BlockDim() == cache_image.info.BlockDim()),
            (image_info.pitch == cache_image.info.pitch),
            (image_info.props.is_pow2 == cache_image.info.props.is_pow2),
            (image_info.alt_tile == cache_image.info.alt_tile),
            (cache_image.info.resources <= image_info.resources), cache_image.info.resources.levels,
            image_info.resources.levels, (cache_image.info.guest_size <= image_info.guest_size),
            expected_size,

            // Size ratios
            static_cast<double>(image_info.guest_size) / expected_size,
            static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

            // Difference between actual and expected sizes with percentages
            static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
            (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

            static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
            (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

            merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
            scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    runtime.CopyMip(&cache_image, &merged_image, mip, slice);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    const auto new_image_id = slot_images.insert(instance, runtime, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    runtime.CopyImage(&src_image, &new_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);
    TrackImage(new_image_id);
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    Vulkan::DrawPipe::CheckOwner("TextureCache::FindImage");
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);

    std::scoped_lock lock{mutex};
    ImageIds image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

    ImageId image_id{};

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_RENDER_PROBLEM(Render_Vulkan, Warning, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        image_id = slot_images.insert(instance, runtime, slot_image_views, info,
                                      desc.type == BindingType::Texture);
        RegisterImage(image_id);
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    ImageIds image_ids;
    ForEachImageWithAddress(address, [&](ImageId image_id, Image& image) {
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_RENDER_PROBLEM(Render_Vulkan, Warning,
                           "Failed to find exact image match for copy addr={:#x}, size={:#x}",
                           address, size);
    }
    return {};
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        image.flags |= ImageFlagBits::GpuModified;
        if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8) &&
            image.info.guest_address != 0) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id);
    return image.FindView(desc.view_info);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateTarget(image_id);

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaType::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaType::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
    UpdateTarget(image_id);

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaType::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    if (desc.info.stencil_addr != 0) {
        ImageId stencil_id{};
        ForEachImageInRegion(
            desc.info.stencil_addr, desc.info.stencil_size, [&](ImageId image_id, Image& image) {
                if (image.info.guest_address != desc.info.stencil_addr) {
                    return;
                }
                if (image.info.pixel_format == vk::Format::eUndefined ||
                    Vulkan::LiverpoolToVK::IsFormatStencilCompatible(image.info.pixel_format)) {
                    stencil_id = image_id;
                }
            });
        if (!stencil_id) {
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id = slot_images.insert(instance, runtime, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        stencil_image.AssociateDepth(image_id, image.image_uid);
    }

    return image.FindView(desc.view_info, false);
}

void TextureCache::RefreshImage(Image& image) {
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));
    const Common::PerfStats::ScopedTimer perf_timer{Common::PerfStats::Id::TextureUploads,
                                                    Common::PerfStats::Id::TextureUploadNs};

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));

        const u32 s_w = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 s_h = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = s_w * s_h * (image.info.num_bits / 8);
        const u64 hash = XXH3_64bits(addr, size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];

        // Protect GPU modified resources from accidental CPU reuploads.
        if (is_gpu_modified && !is_gpu_dirty) {
            const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
            const u64 hash = XXH3_64bits(addr + mip_offset, mip_size);
            if (image.mip_hashes[m] == hash) {
                continue;
            }
            image.mip_hashes[m] = hash;
        }

        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }

    scheduler.EndRendering();
    Common::PerfStats::Add(Common::PerfStats::Id::TextureUploadBytes, image.info.guest_size);
    Common::PerfStats::RecordUploadRegion(image.info.guest_address, image.info.guest_size, true);

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    const auto [buffer, offset] = tile_manager.DetileImage(in_buffer, in_offset, image.info);
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    runtime.UploadImage(&image, buffer, image_copies);
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sampler,
                                     AmdGpu::BorderColorBuffer border_color_base,
                                     const bool is_depth) {
    // Compare and plain uses of one S# need separate samplers.
    const u64 hash = HashCombine(XXH3_64bits(&sampler, sizeof(sampler)), is_depth);

    std::scoped_lock lock{samplers_mutex};
    const auto [it, new_sampler] =
        samplers.try_emplace(hash, instance, sampler, border_color_base, is_depth);
    if (new_sampler) {
        samplers.at(hash).lru_id = sampler_lru_cache.Insert(hash, gc_tick);
    } else {
        sampler_lru_cache.Touch(it->second.lru_id, gc_tick);
    }

    return it->second.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    StampPages(image.info.guest_address, image.info.guest_size,
               image_set_generation.fetch_add(1, std::memory_order_acq_rel) + 1);
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    stat_images.fetch_add(1, std::memory_order_relaxed);
    if (image.info.guest_size > stat_largest_image.load(std::memory_order_relaxed)) {
        stat_largest_image.store(image.info.guest_size, std::memory_order_relaxed);
    }
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    ForEachPage(image.info.guest_address, image.info.guest_size,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    // UpdateTarget: an image can also be freed off the command processor thread
    // outside InvalidateMemory and UnmapMemory (FindImage of a video out buffer
    // registered by the guest)
    StampWrite(image.info.guest_address, image.info.guest_size);
    image.flags &= ~ImageFlagBits::Registered;
    StampPages(image.info.guest_address, image.info.guest_size,
               image_set_generation.fetch_add(1, std::memory_order_acq_rel) + 1);
    if (!lru_cache.Free(image.lru_id)) {
        LOG_ERROR(Render_Vulkan, "Texture cache: LRU item of image addr={:#x} freed twice",
                  image.info.guest_address);
    }
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    stat_images.fetch_sub(1, std::memory_order_relaxed);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == nullptr) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PageShift);
            return;
        }
        auto& image_ids = *page_it;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}", page << PageShift);
            return;
        }
        image_ids.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    if (!image.IsTracked()) {
        // Re-track the whole image
        image.track_addr = image_begin;
        image.track_addr_end = image_end;
        tracker.UpdatePageWatchers(image_begin, image.info.guest_size, PageOp::Track);
    } else {
        if (image_begin < image.track_addr) {
            TrackImageHead(image_id);
        }
        if (image.track_addr_end < image_end) {
            TrackImageTail(image_id);
        }
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    if (image_begin == image.track_addr) {
        return;
    }
    ASSERT(image.track_addr != 0 && image_begin < image.track_addr);
    const auto size = image.track_addr - image_begin;
    image.track_addr = image_begin;
    tracker.UpdatePageWatchers(image_begin, size, PageOp::Track);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_end == image.track_addr_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < image_end);
    const auto addr = image.track_addr_end;
    const auto size = image_end - image.track_addr_end;
    image.track_addr_end = image_end;
    tracker.UpdatePageWatchers(addr, size, PageOp::Track);
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        tracker.UpdatePageWatchers(addr, size, PageOp::Untrack);
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (!image.IsTracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers(image_begin, size, PageOp::Untrack);
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers(addr, size, PageOp::Untrack);
}

void TextureCache::EmergencyCollect() {
    // May be reached from an image upload that already holds the cache lock (FindImage expanding
    // an image): nothing can be freed safely then.
    std::unique_lock lock{mutex, std::try_to_lock};
    if (!lock) {
        LOG_WARNING(Render, "Emergency texture collection skipped, the cache is in use");
        return;
    }
    size_t num_deletions = 512;
    size_t freed = 0;
    const u64 ticks = std::min<u64>(4, gc_tick);
    lru_cache.ForEachItemBelow(gc_tick - ticks, [&](ImageId image_id) {
        if (num_deletions == 0) {
            return true;
        }
        auto& image = slot_images[image_id];
        if (image.SafeToDownload()) {
            if (image.info.IsTiled()) {
                return false; // Can't download non-linear images yet
            }
            DownloadImageMemory(image_id);
        }
        FreeImage(image_id);
        --num_deletions;
        ++freed;
        return false;
    });
    LOG_WARNING(Render, "Emergency texture collection freed {} images", freed);
}

void TextureCache::GarbageCollectImages() {
    const auto now = std::chrono::steady_clock::now();
    gc_level = 0;
    // PINK_GC_TIGHT=0 turns the levels off
    static const bool tight_enabled = [] {
        const char* v = std::getenv("PINK_GC_TIGHT");
        return !v || v[0] != '0';
    }();
    // PINK_GC_HEADROOM=0: the fixed thresholds (from the card size) decide when the collector
    // presses, as before 12/10/2026. With PINK_VRAM_LIMIT_MB at 4500 they sat at 1800 / 2700 MiB,
    // far below the 5-6 GB the game uses, so the collector was always "aggressive": a texture not
    // drawn for 160 submissions (under a second) was freed, and turning the camera back uploaded
    // it again. Now, while the driver reports the free memory, the levels below decide: with
    // 1 GB or more left an image stays until unused for PINK_GC_IDLE_SECONDS (20 s), under 1 GB
    // for 3 s, and only under 640 MiB it goes after a few frames.
    static const bool by_headroom = [] {
        const char* v = std::getenv("PINK_GC_HEADROOM");
        return !v || v[0] != '0';
    }();
    const bool use_headroom = by_headroom && tight_enabled && instance.CanReportMemoryUsage();
    if (instance.CanReportMemoryUsage()) {
        // How much the card still has for this process: what the driver allows less what is in
        // use, where the free space inside the allocator's blocks counts as available (images and
        // buffer arena memory are placed there). The fixed thresholds below know nothing of this:
        // "used" is far above "critical" all the time and the collector frees what was idle for
        // 80 frames whether the card has 2 GB left or nothing. Close to full, images go sooner.
        u64 usage = 0;
        gc_driver_budget = instance.GetDeviceMemoryBudget(&usage);
        total_used_memory = usage;
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(instance.GetAllocator(), budgets);
        const auto& memory_props = instance.GetMemoryProperties();
        u64 allocator_free = 0;
        for (u32 heap = 0; heap < memory_props.memoryHeapCount; ++heap) {
            if (memory_props.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                allocator_free +=
                    budgets[heap].statistics.blockBytes - budgets[heap].statistics.allocationBytes;
            }
        }
        const u64 in_use = usage - std::min(usage, allocator_free);
        gc_headroom = gc_driver_budget > in_use ? gc_driver_budget - in_use : 0;
        if (tight_enabled) {
            gc_level = gc_headroom < 384_MB ? 3 : gc_headroom < 640_MB ? 2 : gc_headroom < 1_GB;
        }
        if (gc_level >= 2 && now - gc_tight_log_time >= std::chrono::seconds{30}) {
            gc_tight_log_time = now;
            LOG_WARNING(Render,
                        "Video memory is tight: {} MiB left of the {} MiB the driver allows ({} "
                        "MiB in use by the emulator, {} MiB of it free inside allocator blocks, "
                        "buffer arenas {} MiB, images {} MiB); freeing images sooner (level {})",
                        gc_headroom >> 20, gc_driver_budget >> 20, usage >> 20,
                        allocator_free >> 20, buffer_cache.ResidentBytes() >> 20,
                        UniqueImage::live_bytes.load() >> 20, gc_level);
        }
    }
    // gc_tick advances once per guest submission, hundreds of times a second: remember it at the
    // start of each second so that an image's age can be compared in seconds (from bbport)
    const u64 second = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count());
    if (second != gc_second) {
        gc_second = second;
        gc_tick_at_second[second % gc_tick_at_second.size()] = gc_tick;
    }
    // PINK_GC_IDLE_SECONDS=<n>: below the pressure mark an image goes only once unused for n seconds
    // (default 20, at most 63) instead of 16 ticks, a few frames, which evicted textures briefly
    // off screen and uploaded them again. 0 restores the 16 ticks.
    static const u64 idle_seconds = [] {
        const char* v = std::getenv("PINK_GC_IDLE_SECONDS");
        return std::min<u64>(v ? std::strtoull(v, nullptr, 10) : 20, 63);
    }();
    // PINK_GC_SKIP_KEPT=0 restores the collector that let images it cannot free use up its deletions
    static const bool skip_kept = [] {
        const char* v = std::getenv("PINK_GC_SKIP_KEPT");
        return !v || std::strtoull(v, nullptr, 10) != 0;
    }();
    const auto report = [&](bool locked) {
        // With SHADPS4_PERF_STATS: what the collector did in the last 10 s
        if (!Common::PerfStats::Enabled() || now - gc_report_time < std::chrono::seconds{10}) {
            return;
        }
        LOG_WARNING(Render,
                    "Image GC: used {} MiB, {} MiB of it buffer arenas (pressure {}, critical {}): "
                    "{} images freed, {} kept (GPU-written) in the last 10 s; driver allows {} "
                    "MiB, {} MiB left, level {}",
                    total_used_memory >> 20, buffer_cache.ResidentBytes() >> 20,
                    pressure_gc_memory >> 20, critical_gc_memory >> 20, gc_freed_images,
                    gc_kept_images, gc_driver_budget >> 20, gc_headroom >> 20, gc_level);
        gc_report_time = now;
        gc_freed_images = gc_kept_images = 0;
        if (!locked) {
            return;
        }
        // Where the video memory is: every registered image by what the collector can do with it
        struct Group {
            u64 count{};
            u64 bytes{};
            void Add(u64 size) {
                ++count;
                bytes += size;
            }
        };
        Group all, kept, linear_gpu, idle, big;
        u64 backings = 0;
        const u64 old_tick = gc_tick - std::min<u64>(160, gc_tick);
        lru_cache.ForEachItemBelow(gc_tick, [&](ImageId image_id) {
            const auto& image = slot_images[image_id];
            u64 size = 0;
            for (const auto& backing : image.backing_images) {
                size += backing.image.size_bytes;
                ++backings;
            }
            all.Add(size);
            const bool gpu_written = image.SafeToDownload();
            if (gpu_written && image.info.IsTiled()) {
                kept.Add(size);
            } else if (gpu_written) {
                linear_gpu.Add(size);
            }
            if (image.lru_touch_tick != ~0ULL && image.lru_touch_tick < old_tick) {
                idle.Add(size);
            }
            if (size >= 32_MB) {
                big.Add(size);
            }
            return false;
        });
        // The allocator's share of video memory: what it took from the driver (blocks) and what
        // is in use inside those blocks (images and buffers); the difference is free space it holds
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(instance.GetAllocator(), budgets);
        const auto& memory_props = instance.GetMemoryProperties();
        u64 vma_blocks = 0;
        u64 vma_allocated = 0;
        u64 driver_usage = 0;
        u64 driver_budget = 0;
        for (u32 heap = 0; heap < memory_props.memoryHeapCount; ++heap) {
            if (!(memory_props.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
                continue;
            }
            vma_blocks += budgets[heap].statistics.blockBytes;
            vma_allocated += budgets[heap].statistics.allocationBytes;
            driver_usage += budgets[heap].usage;
            driver_budget += budgets[heap].budget;
        }
        const auto [ring_bytes, large_bytes] =
            runtime.GetStagingPool().HeldBytes(MemoryType::DeviceLocal);
        LOG_WARNING(Render,
                    "Video memory: driver reports {} of {} MiB; allocator blocks {} MiB with {} "
                    "MiB in use (images {} MiB, the rest buffers); device-local staging: ring {} "
                    "MiB, dedicated {} MiB; buffer arenas {} MiB",
                    driver_usage >> 20, driver_budget >> 20, vma_blocks >> 20, vma_allocated >> 20,
                    UniqueImage::live_bytes.load() >> 20, ring_bytes >> 20, large_bytes >> 20,
                    buffer_cache.ResidentBytes() >> 20);
        LOG_WARNING(Render,
                    "Image GC detail: host images {} = {} MiB alive; "
                    "registered {} = {} MiB in {} backings; GPU-written tiled (cannot be freed) "
                    "{} = {} MiB; GPU-written linear {} = {} MiB; unused for 160 ticks {} = {} "
                    "MiB; 32 MiB or larger {} = {} MiB; tick {}",
                    UniqueImage::live_count.load(), UniqueImage::live_bytes.load() >> 20, all.count, all.bytes >> 20, backings, kept.count,
                    kept.bytes >> 20, linear_gpu.count, linear_gpu.bytes >> 20, idle.count,
                    idle.bytes >> 20, big.count, big.bytes >> 20, gc_tick);
    };
    if (total_used_memory < trigger_gc_memory) {
        report(false);
        return;
    }
    // The tick at the start of that second (an older one when no submission came then)
    const u64 idle_tick =
        idle_seconds != 0
            ? gc_tick_at_second[(second - idle_seconds) % gc_tick_at_second.size()]
            : ~0ULL;
    std::scoped_lock lock{mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 below_tick = 0;
    size_t num_deletions = 0;
    u32 visited = 0;

    // The tick at the start of the second 3 s ago (level 1 with use_headroom)
    const u64 short_idle_tick = gc_tick_at_second[(second - 3) % gc_tick_at_second.size()];
    const auto over_critical = [&] {
        return use_headroom ? gc_level >= 2 : total_used_memory >= critical_gc_memory;
    };
    const auto configure = [&](bool allow_aggressive) {
        pressured = use_headroom ? gc_level >= 1 : total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && over_critical();
        static constexpr std::array<u64, 4> TightTicks = {~0ULL, 48, 24, 8};
        const u64 ticks_to_destroy = std::min<u64>(
            {u64{aggresive ? 160u : pressured ? 80u : 16u}, TightTicks[gc_level], gc_tick});
        below_tick = gc_tick - ticks_to_destroy;
        if (!pressured && !aggresive && gc_level == 0) {
            below_tick = std::min(below_tick, idle_tick);
        } else if (use_headroom && gc_level == 1) {
            below_tick = std::min(below_tick, short_idle_tick);
        }
        num_deletions = aggresive ? (small_vram ? 128 : 40) : pressured ? 20 : 10;
        if (gc_level != 0) {
            num_deletions = std::max<size_t>(num_deletions, 64u << gc_level);
        }
        visited = 0;
    };
    const auto clean_up = [&](ImageId image_id) {
        if (num_deletions == 0 || ++visited > (gc_level >= 2 ? 2048u : 512u)) {
            return true;
        }
        auto& image = slot_images[image_id];
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        // GPU-written images cannot go now: tiled ones cannot be written back at all (a workaround
        // for now, non-linear image downloads are not handled), linear ones only under pressure.
        // From bbport: they neither use up the deletions nor stay first in line. With the oldest
        // images all being such render targets the collector freed nothing and VRAM only grew.
        if ((tiled && download) || (download && !pressured)) {
            if (skip_kept) {
                TouchImage(image);
            } else {
                --num_deletions;
            }
            ++gc_kept_images;
            return false;
        }
        --num_deletions;
        if (download) {
            DownloadImageMemory(image_id);
        }
        FreeImage(image_id);
        ++gc_freed_images;
        if (!use_headroom && total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    lru_cache.ForEachItemBelow(below_tick, clean_up);

    if (over_critical()) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        lru_cache.ForEachItemBelow(below_tick, clean_up);
    }
    report(true);
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    std::scoped_lock lock{samplers_mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](u64 hash) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        const size_t lru_id = samplers.at(hash).lru_id;
        samplers.erase(hash);
        sampler_lru_cache.Free(lru_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    Vulkan::DrawPipe::CheckOwner("TextureCache::RunGarbageCollector");
    SCOPE_EXIT {
        ++gc_tick;
    };

    GarbageCollectImages();
    GarbageCollectSamplers();
}

void TextureCache::TouchImage(Image& image) {
    if (lru_cache.Touch(image.lru_id, gc_tick)) {
        image.lru_touch_tick = gc_tick;
    } else {
        // The image was already freed, so leave its LRU entry alone
        // Adding it back to the list could create a cycle and keep garbage collection running
        // forever
        static std::atomic<u32> reports{};
        if (reports.fetch_add(1, std::memory_order_relaxed) < 20) {
            LOG_ERROR(Render_Vulkan,
                      "Texture cache: touch of a freed image addr={:#x} size={:#x} flags={:#x}",
                      image.info.guest_address, image.info.guest_size,
                      static_cast<u32>(image.flags));
        }
    }
}

void TextureCache::EnablePageStamps() {
    // Use 8 MB to keep one generation stamp for each 1 MB page in the 40-bit GPU address space
    page_stamps = std::make_unique<std::atomic<u64>[]>(NumStampPages);
}

void TextureCache::StampPages(VAddr addr, u64 size, u64 generation) {
    if (!page_stamps) {
        return;
    }
    // Stamp the same pages used to register this image so cached lookups can see every change that
    // affects their search range
    ForEachPage(addr, size, [&](u64 page) {
        if (page < NumStampPages) {
            page_stamps[page].store(generation, std::memory_order_release);
        }
    });
}

void TextureCache::StampWrite(VAddr addr, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 generation = write_generation.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 page_end = std::min((addr + size - 1) >> Traits::PAGE_BITS, NumStampPages - 1);
    for (u64 page = addr >> Traits::PAGE_BITS; page <= page_end; ++page) {
        write_stamps[page].store(generation, std::memory_order_relaxed);
    }
    // The stamps go before the image fields the caller changes next (UpdateTarget reads
    // them the other way round)
    std::atomic_thread_fence(std::memory_order_release);
}

bool TextureCache::PagesWrittenSince(VAddr addr, u64 size, u64 generation) const noexcept {
    const u64 page_end = (addr + size - 1) >> Traits::PAGE_BITS;
    if (size == 0 || page_end >= NumStampPages) {
        return true;
    }
    for (u64 page = addr >> Traits::PAGE_BITS; page <= page_end; ++page) {
        if (write_stamps[page].load(std::memory_order_relaxed) > generation) {
            return true;
        }
    }
    return false;
}

void TextureCache::UpdateTarget(ImageId image_id) {
    const Image& image = slot_images[image_id];
    const VAddr begin = image.info.guest_address;
    const VAddr end = begin + image.info.guest_size;
    const auto unchanged = [&](const Image& img) {
        return True(img.flags & ImageFlagBits::Registered) &&
               False(img.flags & ImageFlagBits::Dirty) && img.track_addr == begin &&
               img.track_addr_end == end && img.lru_touch_tick == gc_tick;
    };
    // Read image fields before page stamps when checking an update without the lock
    // Guest writes stamp the pages first, so a concurrent change rejects the skip
    if (image.target_write_gen != 0 && Common::PerfStats::AbToggleOn(target_fast_toggle) &&
        unchanged(image)) {
        std::atomic_thread_fence(std::memory_order_acquire);
        if (!PagesWrittenSince(begin, end - begin, image.target_write_gen)) {
            Common::PerfStats::Add(Common::PerfStats::Id::TargetUpdatesSkipped);
            if (!target_fast_verify) {
                return;
            }
            std::scoped_lock lk{mutex};
            if (const Image& locked = slot_images[image_id]; !unchanged(locked)) {
                Common::PerfStats::Add(Common::PerfStats::Id::TargetSkipMismatches);
                static std::atomic<u32> reports{};
                if (reports.fetch_add(1, std::memory_order_relaxed) < 20) {
                    LOG_ERROR(Render_Vulkan,
                              "Target fast verify: skipped the update of addr={:#x} size={:#x} "
                              "flags={:#x} track={:#x}-{:#x} touch={} gc_tick={}",
                              begin, end - begin, static_cast<u32>(locked.flags),
                              locked.track_addr, locked.track_addr_end, locked.lru_touch_tick,
                              gc_tick);
                }
            }
        }
    }
    std::scoped_lock lk{mutex};
    // Fetch the image again under the lock
    // Guest video buffer registration can grow slot_images while this thread waits
    Image& locked = slot_images[image_id];
    TrackImage(image_id);
    TouchImage(locked);
    RefreshImage(locked);
    // Writers bump the generation under the lock: any later one stamps a newer value
    locked.target_write_gen = write_generation.load(std::memory_order_relaxed);
}

void TextureCache::TouchReusedImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    if (image.lru_touch_tick != gc_tick) {
        std::scoped_lock lock{mutex};
        if (True(image.flags & ImageFlagBits::Registered)) {
            TouchImage(image);
        }
    }
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(!image.IsTracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.erase(image_view_id);
            }
        }
        slot_images.erase(image_id);
    });
}

} // namespace VideoCore
