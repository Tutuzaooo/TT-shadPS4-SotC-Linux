// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <chrono>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/mapped_page_table.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_bind_cache.h"
#include "video_core/renderer_vulkan/vk_draw_packet.h"
#include "video_core/renderer_vulkan/vk_gpu_checkpoints.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class ComputePipeline;
class GraphicsPipeline;
class Runtime;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    /// gcn_unordered_dispatches: ring of the packets processed next (0 = graphics,
    /// 1 + vqid = compute queue)
    void SetCommandRing(u8 ring) noexcept {
        command_ring = ring < MaxCommandRings ? ring : MaxCommandRings - 1;
    }
    /// gcn_unordered_dispatches: the game sent a sync packet (event write,
    /// EOP/EOS/RELEASE_MEM, ACQUIRE_MEM, a wait) on the current ring
    void OnGameSync();

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    /// Per-game lod_stats_from_bindings: answers IT_GET_LOD_STATS from the counter banks of the T#s bound
    /// since the previous packet. Returns false when the workaround is off
    bool WriteLodStats(VAddr address, u32 num_bytes);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    void ReleaseCpuAuthoritativeRange(VAddr addr, u64 size);
    bool IsMapped(VAddr addr, u64 size);
    /// Clean reads (srt_walker_clean_reads, shader_code_clean_reads): copies GPU-mapped bytes the GPU never
    /// wrote from the physical backing without touching their page, which may be protected because it shares
    /// a GPU-written buffer. A readback would leave the same value there. Returns false for GPU-written or
    /// unbacked bytes. Command processor thread only
    bool ReadCleanMemory(VAddr addr, void* out, u64 size);

    /// Returns false for a T# that describes an image Vulkan cannot create (garbage descriptor)
    bool IsPlausibleImage(const VideoCore::ImageInfo& info, bool log = true);
    /// Check whether the texture can be bound, or should use a null descriptor
    /// Share the checks with descriptor arrays and return the built bound_desc
    /// Enable log to report a rejection
    bool CheckBindableImage(const AmdGpu::Image& tsharp, const Shader::ImageResource& image_desc,
                            std::optional<VideoCore::TextureCache::ImageDesc>& bound_desc,
                            bool log);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();
    void OnFence();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    bool IsMappedCached(VAddr addr, VAddr end);
    /// With SHADPS4_MAPPED_PAGE_TABLE_VERIFY, compare each table answer with the interval set and
    /// report how often the table returns Unknown once a minute
    void VerifyMappedPageTable(VAddr addr, u64 size, VideoCore::MappedPageTable::Answer answer);
    /// Mirror the mapping change for [addr, addr + size) in the page table while
    /// mapped_ranges_mutex is still held
    void UpdateMappedPageTable(VAddr addr, u64 size, bool mapped);
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    /// Look up targets for PrepareRenderState; return false if FindImage creates an image or resolves overlaps
    bool PrepareRenderStateFull(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    RenderState BeginRenderingFull(const GraphicsPipeline* pipeline);
    /// Layout BeginRendering gives the current draw's depth target, none without one
    std::optional<vk::ImageLayout> DepthTargetLayout(const VideoCore::Image& image) const;
    static bool IsReadOnlyDepthLayout(vk::ImageLayout layout);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    /// Describe a replacement operation for this draw using the current stage's registers; None keeps the draw, and copy flags apply to
    /// DepthStencilCopy
    DrawPacket::Filter FilterDraw(u8& depth_copy, u8& stencil_copy) const;
    void RunFilter(const DrawPacket::Header& packet);

    // Stage A filters the draw, selects its pipeline and builds a packet; stage B records it immediately afterward on the command processor
    // thread, using Regs(), extent accessors and Shader::Info accessors to read the packet's state
    void DrawPrepare(bool is_indexed, u32 index_offset);
    void DrawIndirectPrepare(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                             u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                             u16 instance_sgpr_offset);
    void DispatchPrepare(bool indirect, VAddr address, u32 size, s8 ring);
    /// Write the packet header, changed register blocks, target extents and each pipeline stage's user data
    DrawPacket::Header& BuildPacket(DrawPacket::Kind kind, const Pipeline* pipeline, bool compute);
    void RunPacket();
    /// SHADPS4_PIPE_VERIFY=N compares stage B's captured state with live state every N packets
    void VerifyPacket(const DrawPacket::Header& packet);
    void DrawRecord(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset);
    void DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed, VAddr arg_address,
                            u32 stride, u32 max_count, VAddr count_address);
    void DispatchRecord(const ComputePipeline* pipeline);
    void DispatchIndirectRecord(const ComputePipeline* pipeline, VAddr address, u32 size);
    const AmdGpu::Regs& Regs() const;
    AmdGpu::CbDbExtent CbExtent(u32 index) const;
    AmdGpu::CbDbExtent DbExtent() const;
    const AmdGpu::ComputeProgram& CsRegs() const;

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    /// dynamic_tsharp_array_size: logs the first descriptor array bound, then a per-minute count
    void ReportTsharpArray(u64 pgm_hash, u32 num_elements, u32 null_elements);
    /// SHADPS4_CLEAN_READ_VERIFY: reads back the bytes a clean read served and reports any difference
    void VerifyCleanRead(VAddr addr, const void* served, u64 size);
    bool BindResources(const Pipeline* pipeline);
    /// Refreshes sampled descriptors with their final subresource layouts
    void RefreshImageDescriptorLayouts();

    /// gpu_checkpoints: attaches SHADPS4_GPU_DIAG details to the command just recorded and runs the
    /// SHADPS4_GPU_SYNC_EVERY bisection
    void OnGpuCommandRecorded();
    /// SHADPS4_GPU_DIAG: copies words of a buffer into host memory on the GPU, right before the command that
    /// reads them. Returns null when no capture buffer exists
    const u32* CaptureWords(vk::Buffer buffer, u64 offset, u32 num_words);
    void CaptureBufferHeads();

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0);

    void ResetBindings(bool is_compute);
    /// Count reused V# and T# slots checked against a full binding and log any mismatch so the
    /// cache assumptions can be verified
    void VerifyIncrementalBind(bool matches, const char* kind, u64 pgm_hash, VAddr address);
    /// Submits periodically without waiting
    void FlushPeriodic();

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    // Allow descriptor arrays in every stage, with array_size + 1 bindings per array and NUM_IMAGE_ARRAY_DESCRIPTORS limiting each shader;
    // image_infos also holds samplers, and bound_images also holds targets, including rebound targets
    static constexpr u32 MaxImageDescriptors =
        Shader::NUM_IMAGES + Shader::NUM_IMAGE_ARRAY_DESCRIPTORS;
    static constexpr u32 MaxPipelineImageDescriptors =
        (MaxImageDescriptors + Shader::NUM_SAMPLERS) * Shader::MaxStageTypes +
        2 * AmdGpu::NUM_COLOR_BUFFERS + 1;
    boost::container::static_vector<vk::DescriptorImageInfo, MaxPipelineImageDescriptors>
        image_infos;
    // Sampled descriptors that need a final layout refresh
    struct ImageDescriptorRef {
        u32 info_index;
        const VideoCore::Image::BackingImage* backing;
        u32 subres_idx;
    };
    boost::container::static_vector<ImageDescriptorRef, MaxPipelineImageDescriptors>
        image_descriptor_refs;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, MaxPipelineImageDescriptors> bound_images;
    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        bool is_written;
    };
    boost::container::static_vector<BoundBuffer, Shader::NUM_BUFFERS> bound_buffers;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, MaxImageDescriptors> image_bindings;
    bool attachment_feedback_loop{};
    bool needs_barrier{};
    // gpu_checkpoints diagnostics (SHADPS4_GPU_DIAG, SHADPS4_GPU_SYNC_EVERY)
    GpuCheckpoints::Detail diag_detail{};
    std::array<vk::DescriptorBufferInfo, GpuCheckpoints::Detail::MaxBuffers> diag_sources{};
    static constexpr u64 DiagCaptureBufferSize = 8_MB;
    std::unique_ptr<VideoCore::Buffer> diag_capture_buffer;
    u64 diag_capture_offset{};
    u32 diag_commands_since_sync{};
    std::chrono::steady_clock::time_point diag_start{std::chrono::steady_clock::now()};
    // lod_stats_from_bindings: bindings per T# counter_bank_id since the last IT_GET_LOD_STATS. Only the GPU
    // command processor thread binds textures and processes the packet
    const bool lod_stats_enabled;
    const u32 periodic_flush_commands;
    // cp_recording_cuts: IsMapped answers from a per-thread cache of mapped runs
    const bool recording_cuts;
    const bool recording_cuts_toggle; ///< cp_recording_cuts follows SHADPS4_AB_TOGGLE
    // Keep a table of 16 KB mapped pages beside mapped_ranges for checks without the shared lock,
    // or leave it null when the setting is off
    std::unique_ptr<VideoCore::MappedPageTable> mapped_page_table;
    const bool mapped_page_table_toggle; ///< Whether mapped_page_table is active for this interval of SHADPS4_AB_TOGGLE
    const bool mapped_page_table_verify;
    // Keep the cache for the pipeline being bound, its starting generations and the next V# and T#
    // slots to visit
    // The cache pointer is null when incremental_bind is disabled so the call follows the regular
    // binding path
    const bool incremental_bind;
    const bool incremental_bind_toggle; ///< Whether incremental_bind is active for this interval of SHADPS4_AB_TOGGLE
    /// With SHADPS4_INCREMENTAL_BIND_VERIFY, bind reusable slots through the full path as well and
    /// compare the two results
    const bool incremental_bind_verify;
    BindCache* bind_cache{};
    BindCache::Generations bind_generations{};
    u32 bind_cache_buffer_index{};
    u32 bind_cache_image_index{};
    // Share T# resolutions across pipelines and remember whether that cache is enabled for the
    // current binding call
    // The pointer stays null when tsharp_cache is disabled and the regular resolution path remains
    // available
    std::unique_ptr<TsharpCache> tsharp_cache;
    const bool tsharp_cache_toggle; ///< Whether tsharp_cache is active for this interval of SHADPS4_AB_TOGGLE
    /// With SHADPS4_TSHARP_CACHE_VERIFY, resolve cached T# entries through the full lookup again
    /// and compare the images and views
    const bool tsharp_cache_verify;
    bool tsharp_cache_on{};
    // depth_target_sampled_layout, whether it follows SHADPS4_AB_TOGGLE and whether it
    // is on for the draw being recorded
    const bool depth_target_sampled_layout;
    const bool depth_target_sampled_toggle;
    bool depth_target_shared{};
    static constexpr vk::AccessFlags2 SharedDepthTargetAccess =
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
        vk::AccessFlagBits2::eDepthStencilAttachmentRead;

    /// gcn_unordered_dispatches: notes a guest draw or dispatch on the current ring
    /// For a dispatch whose previous command on the ring was a dispatch, with
    /// no sync packet since, returns the ring (GCN may start it before the
    /// previous one ends), else -1
    s8 NoteRingCommand(bool is_dispatch);
    static constexpr u8 MaxCommandRings = 16;
    const bool gcn_unordered_dispatches;
    const bool gcn_unordered_toggle; ///< gcn_unordered_dispatches follows SHADPS4_AB_TOGGLE
    u8 command_ring{};
    std::array<bool, MaxCommandRings> ring_last_dispatch{};
    std::array<bool, MaxCommandRings> ring_synced{};
    // Of the dispatch being recorded: its NoteRingCommand, and whether its buffer
    // accesses are those of a game dispatch without DMA (DMA reaches memory the
    // barrier tracking does not see)
    s8 unordered_ring{-1};
    bool game_dispatch_accesses{};
    const u64 unbounded_cap_bytes; ///< unbounded_vsharp_cap_mb in bytes, 0 = off

    // Cache targets while CB/DB registers and target-page images stay the same, and cache render state while targets and metadata stay the
    // same; exclude clears and feedback loops
    struct RsTargetsKey {
        static constexpr size_t MaxBytes =
            AmdGpu::NUM_COLOR_BUFFERS * (sizeof(AmdGpu::ColorBuffer) + sizeof(AmdGpu::CbDbExtent)) +
            sizeof(AmdGpu::DepthBuffer) + sizeof(AmdGpu::DepthView) + sizeof(AmdGpu::CbDbExtent) +
            64;
        std::array<u8, MaxBytes> bytes;
        u32 size;
        bool operator==(const RsTargetsKey& other) const noexcept;
    };
    struct RsTargetsMemo {
        bool valid{};
        RsTargetsKey key;
        u64 image_sets{};
        u32 num_cbs{};
        std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
        std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    };
    struct RsBeginSig {
        static constexpr u32 NumTargets = AmdGpu::NUM_COLOR_BUFFERS + 1;
        std::array<VideoCore::ImageId, NumTargets> ids{};
        std::array<u64, NumTargets> uids{};
        std::array<VideoCore::ImageViewInfo, NumTargets> views{};
        std::array<VAddr, NumTargets> metas{};
        std::array<u8, AmdGpu::NUM_COLOR_BUFFERS> color_samples{};
        u32 mrt_mask{};
        u64 meta_generation{};
        bool depth_clear{};
        bool stencil_clear{};
        bool depth_valid{};
        bool stencil_valid{};
        bool depth_shared{};
        bool operator==(const RsBeginSig&) const = default;
    };
    struct RsBeginMemo {
        bool valid{};
        RsBeginSig sig;
        RenderState state;
        std::array<const VideoCore::Image::BackingImage*, RsBeginSig::NumTargets> backings{};
    };
    using RsBackings = decltype(RsBeginMemo::backings);
    void MakeTargetsKey(const GraphicsPipeline* pipeline, RsTargetsKey& out) const;
    bool TargetsUnchanged(const RsTargetsMemo& memo) const;
    void ReuseTargets(const RsTargetsMemo& memo);
    RsBeginSig MakeBeginSig(const GraphicsPipeline* pipeline);
    /// Check whether the full path would produce the cached state for these targets
    bool BeginMemoHolds(const RsBeginSig& sig);
    /// Return the cached state after per-draw updates, or none if an update changed a backing
    std::optional<RenderState> ReuseBeginState();
    void StoreBeginMemo(const GraphicsPipeline* pipeline, const RenderState& state);
    RsBackings TargetBackings(const RsBeginSig& sig);
    /// SHADPS4_RS_MEMO_VERIFY counts checks, logs the first mismatches and reports totals once a minute
    void VerifyRenderStateMemo(bool matches, const GraphicsPipeline* pipeline, bool targets,
                               const RenderState* memo, const RenderState* full);
    const bool rs_memo;
    const bool rs_memo_toggle; ///< Follow SHADPS4_AB_TOGGLE for render_state_memo
    /// Rebuild reused results through the full path during SHADPS4_RS_MEMO_VERIFY and draw with that result
    const bool rs_memo_verify;
    bool rs_memo_on{}; ///< State for the draw being recorded
    RsTargetsMemo rs_targets;
    RsBeginMemo rs_begin;
    u64 rs_verify_draws{};
    u64 rs_verify_targets{}; ///< Reused target lookups
    u64 rs_verify_states{};  ///< Reused render states
    u64 rs_verify_target_mismatches{};
    u64 rs_verify_state_mismatches{};
    std::chrono::steady_clock::time_point rs_verify_last_report{};

    u64 bind_verify_checks{};
    u64 bind_verify_mismatches{};
    u64 bind_verify_raced{};
    std::chrono::steady_clock::time_point bind_verify_last_report{};
    std::array<u32, 256> lod_stats_uses{};
    u64 lod_stats_packets_minute{};
    u64 lod_stats_banks_minute{};
    std::chrono::steady_clock::time_point lod_stats_last_report{};
    // dynamic_tsharp_array_size: descriptor arrays bound and their null elements
    u64 tsharp_array_binds_minute{};
    u64 tsharp_array_null_minute{};
    std::chrono::steady_clock::time_point tsharp_array_last_report{};
    // Store the growing, 8-byte aligned packet buffer, stage B's registers and extents, and the active packet view; outside recording,
    // accessors use live state
    std::unique_ptr<u64[]> packet_buffer;
    u32 packet_capacity{};
    std::unique_ptr<AmdGpu::Regs> pipe_regs;
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> pipe_cb_extent{};
    AmdGpu::CbDbExtent pipe_db_extent{};
    static inline thread_local const AmdGpu::Regs* stage_regs = nullptr;
    static inline thread_local const AmdGpu::ComputeProgram* stage_cs = nullptr;
    const u32 pipe_verify_interval; ///< SHADPS4_PIPE_VERIFY interval; 0 disables checks
    u64 pipe_verify_packets{};
    u64 pipe_verify_checks{};
    u64 pipe_verify_mismatches{};
    std::chrono::steady_clock::time_point pipe_verify_last_report{};
};

} // namespace Vulkan
