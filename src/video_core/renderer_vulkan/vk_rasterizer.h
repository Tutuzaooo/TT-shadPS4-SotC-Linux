// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <unordered_set>
#include <vector>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/mapped_page_table.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_bind_cache.h"
#include "video_core/renderer_vulkan/vk_draw_pipe.h"
#include "video_core/renderer_vulkan/vk_draw_prep.h"
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
    /// After InvalidateMemory handled a guest write fault at `addr` (see BufferCache::ExtendWriteFault)
    void ExtendWriteFault(VAddr addr);
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

    /// Draw preparation helpers (vk_draw_prep.h), null when off
    DrawPreparation* GetDrawPreparation() {
        return draw_prep.get();
    }
    /// Command processor thread: hash of the register state as it is now (draw preparation)
    u64 CurrentStateHash();

    /// Draw pipe (vk_draw_pipe.h): on the command processor thread, waits until the draw
    /// recording thread has run every draw handed to it. That thread owns the caches, the
    /// scheduler and the runtime while draws are in flight, so everything else the command
    /// processor thread does on them comes here first. No-op without the pipe, with the pipe idle
    /// and on other threads
    void DrainDrawPipe(u32 reason = DrawPipe::ReasonRasterizer, u32 detail = 0) {
        if (draw_pipe) [[unlikely]] {
            DrainDrawPipeSlow(reason, detail);
        }
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

public:
    /// Draw pipe: runs `task(context, data)` in order with the draws handed to the recording
    /// thread: queued for that thread (from pipe mode `min_mode` on; `data` is copied), or here
    /// after waiting for it. `reason` and `detail` label that wait in the statistics
    using OrderedTask = void (*)(void* context, const u8* data);
    /// Returns true when the task was queued (it has not run yet)
    bool RunInOrder(OrderedTask task, void* context, const void* data, u32 size, u32 min_mode,
                    u32 reason, u32 detail = 0);
    /// Draw pipe: the position after the last queued packet, and whether the recording thread
    /// has run everything up to a position
    [[nodiscard]] u64 DrawPipeHead() const {
        return draw_pipe ? draw_pipe->Head() : 0;
    }
    [[nodiscard]] bool DrawPipeReached(u64 position) const {
        return !draw_pipe || draw_pipe->Reached(position);
    }
    /// PINK_DRAW_PIPE is at least `mode`
    [[nodiscard]] bool DrawPipeModeAtLeast(u32 mode) const noexcept {
        return static_cast<u32>(pipe_mode) >= mode;
    }

    /// PINK_PIPE_EXTRA=<list>: the pieces of pipe mode 7, each on its own on top of mode 6, so
    /// the one behind a defect can be found (mode 7 turns the first three on together)
    enum PipeExtra : u32 {
        PipeExtraEos = 1,     ///< "eos": EVENT_WRITE_EOS queued in order
        PipeExtraAsc = 2,     ///< "asc": compute queue RELEASE_MEM and read pointer in order
        PipeExtraWait = 4,    ///< "wait": graphics WAIT_REG_MEM on a queued label counts as met
        PipeExtraAscWait = 8, ///< "ascwait": compute WAIT_REG_MEM likewise, and no wait when it
                              ///< already passes
    };
    /// The pipe mode a piece starts at: 6 with its PINK_PIPE_EXTRA flag, `mode_without` without
    [[nodiscard]] u32 PipeExtraMode(u32 extra, u32 mode_without) const noexcept {
        return (pipe_extra & extra) ? 6u : mode_without;
    }
    /// Stage A: whether a WAIT_REG_MEM that a queued label meets may go on before stage B has
    /// run the packets queued before that label. Stage A then reads guest memory (shader resource
    /// tables) that those packets may still write: it can do so only while it knows what they
    /// write (pending writes, below) and none of them writes memory it cannot bound
    [[nodiscard]] bool OrderedWaitSafe() const;
    void NoteOrderedWaitSkipped() noexcept {
        ++pipe_waits_skipped;
    }

private:
    /// Draw pipe, PINK_DRAW_PIPE=<n>. Each mode adds one thing to the previous one:
    /// 0 off (default): draws run as before.
    /// 1 packets on the same thread: a draw is written as a packet (changed register blocks, user
    ///   data of its stages, parameters) and run from it at once. Tests that the packet carries
    ///   everything the second half of a draw reads.
    /// 2 lockstep: the packet runs on the draw recording thread while the command processor
    ///   thread waits for it. Tests the second thread (faults, command recorder) without overlap.
    /// 3 draws overlap: the command processor thread goes on decoding while draws are recorded;
    ///   it waits before dispatches and everything else.
    /// 4 dispatches are queued like draws.
    /// 5 end-of-pipe events (fences, labels) are queued in order with the draws.
    /// 6 fewer waits: the end of a submission is queued in order too, a WAIT_REG_MEM that
    ///   already passes and EVENT_WRITE do not wait.
    /// 7 a WAIT_REG_MEM on a label a queued fence will write counts as met; end-of-shader events
    ///   and the compute queues' fences and read pointers are queued in order.
    enum class PipeMode : u32 {
        Off,
        Inline,
        Lockstep,
        Draws,
        DrawsAndDispatches,
        Fences,
        FewerWaits,
        QueuedFences,
    };
    enum class PacketKind : u32 { Draw, DrawIndirect, Dispatch, DispatchIndirect, Special, Task };
    struct PipeIndirect {
        VAddr address;
        VAddr count_address;
        u32 offset;
        u32 stride; ///< size of the arguments for a dispatch
        u32 max_count;
        u32 direct; ///< direct_args holds the draw's arguments: record a direct draw
        std::array<u32, 5> direct_args;
    };
    [[nodiscard]] bool UseDrawPipe() const noexcept {
        return pipe_mode != PipeMode::Off;
    }
    [[nodiscard]] bool OnStageA() const;
    void DrainDrawPipeSlow(u32 reason, u32 detail);
    /// Writes the packet of a draw or dispatch and hands it to the second half (PipeMode)
    void PostPacket(PacketKind kind, const Pipeline* pipeline, bool is_indexed, u32 index_offset,
                    const PipeIndirect* indirect);
    static void RunDrawPacket(void* context, const u8* data, u32 size);
    void PrintPipeStats();
    /// FilterDraw's decision without its side effects
    [[nodiscard]] bool FilterDrawPasses() const;
    /// The second half of a draw or dispatch: everything after the pipeline was selected
    void DrawRecord(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset);
    void DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed, VAddr arg_address,
                            u32 offset, u32 stride, u32 max_count, VAddr count_address,
                            const u32* direct_args = nullptr);
    /// PINK_DIRECT_DRAWS: the arguments of a single indirect draw when they can be read from
    /// guest memory now, in command order (not GPU-written, no queued packet writes them)
    bool ReadDirectDrawArgs(VAddr addr, u32 size, u32* out);
    void DispatchRecord(const ComputePipeline* pipeline);
    void DispatchIndirectRecord(const ComputePipeline* pipeline, VAddr address, u32 offset,
                                u32 size);

    /// Registers as the current draw sees them: the copy the running packet was applied to while
    /// one runs (stage_regs), the command processor's otherwise
    [[nodiscard]] const AmdGpu::Regs& DrawRegs() const noexcept;
    [[nodiscard]] const AmdGpu::ComputeProgram& DrawCsRegs() const noexcept;
    [[nodiscard]] const AmdGpu::CbDbExtent& CbExtent(u32 index) const noexcept;
    [[nodiscard]] const AmdGpu::CbDbExtent& DbExtent() const noexcept;
    static inline thread_local const AmdGpu::Regs* stage_regs = nullptr;
    static inline thread_local const AmdGpu::ComputeProgram* stage_cs = nullptr;
    static inline thread_local const AmdGpu::CbDbExtent* stage_cb_extent = nullptr;
    static inline thread_local const AmdGpu::CbDbExtent* stage_db_extent = nullptr;

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

    bool FilterDraw();

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

    /// PINK_VERTEX_MEMO (experimental build, default on): the last vertex buffer bindings of each
    /// pipeline and the last index buffer binding, reused while the guest buffers, the sync batch,
    /// the GPU mappings and the arenas are unchanged (as incremental_bind does for V#s)
    bool vertex_memo{};
    struct VertexMemo {
        const void* pipeline{};
        u64 sync_flushes{~0ULL};
        u64 maps{};
        u64 arenas{};
        VertexInputs<AmdGpu::Buffer> guest;
        VertexInputs<vk::Buffer> host_buffers;
        VertexInputs<vk::DeviceSize> host_offsets;
        VertexInputs<vk::DeviceSize> host_sizes;
        VertexInputs<vk::DeviceSize> host_strides;
        struct Range {
            const VideoCore::Buffer* buffer;
            u64 offset;
            u64 size;
        };
        VertexInputs<Range> ranges;
    };
    std::array<VertexMemo, 256> vertex_memo_table{};
    struct IndexMemo {
        VAddr address{};
        u32 size{};
        u64 sync_flushes{~0ULL};
        u64 maps{};
        u64 arenas{};
        const VideoCore::Buffer* buffer{};
        u64 offset{};
    } index_memo{};
    u64 vertex_memo_hits{}, vertex_memo_misses{}, index_memo_hits{}, index_memo_misses{};
    /// Written bindings with no image at their address, while no image was registered or removed
    /// since (experimental build): the texture cache lookup is skipped
    struct NoImageMemo {
        VAddr address{};
        u64 size{};
        u64 image_sets{~0ULL};
    };
    std::array<NoImageMemo, 1024> no_image_memo{};

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
    // Each stage can bind NUM_BUFFERS entries; buffer_infos and bound_buffers need room for every stage, like the image lists
    static constexpr u32 MaxPipelineBuffers = Shader::NUM_BUFFERS * Shader::MaxStageTypes;
    boost::container::static_vector<vk::DescriptorBufferInfo, MaxPipelineBuffers> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, MaxPipelineImageDescriptors> bound_images;
    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        bool is_written;
    };
    boost::container::static_vector<BoundBuffer, MaxPipelineBuffers> bound_buffers;

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
    /// PINK_STREAM_REUSE (default on): small read-only buffers and flattened user data whose bytes
    /// match their last copy into the stream buffer in the same command buffer bind that copy again
    bool stream_reuse{};
    /// PINK_STREAM_REUSE_TICKS (experimental build, default on): reuse across command buffers
    /// while fewer than this many bytes went into the stream buffer since the copy (0: off)
    u64 stream_reuse_distance{};
    struct FlatMemo {
        const void* stage{};
        u64 tick{~0ULL};
        u64 pos{~0ULL};
        u64 offset{};
        std::vector<u32> data;
    };
    std::array<FlatMemo, 512> flat_memo{};
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

    // Draw pipe
    PipeMode pipe_mode{PipeMode::Off};
    u32 pipe_verify_interval{};
    bool pipe_synced{};                       ///< pipe_regs holds the registers of the last packet
    std::unique_ptr<AmdGpu::Regs> pipe_regs;  ///< the second half's copy of the register file
    std::vector<u8> inline_packet;            ///< PipeMode::Inline: the packet being run
    u64 pipe_packets{};
    std::array<u64, 6> pipe_packets_by_kind{};
    u64 pipe_stats_packets{};
    u64 pipe_stats_busy{};
    u64 pipe_stats_tsc{};
    std::chrono::steady_clock::time_point pipe_stats_time{};
    // Draw preparation: the register state hash, and the blocks written since it was updated
    RegStateHash state_hash;
    Common::BlockBits<AmdGpu::RegDirty::NumBlocks> hash_pending;
    bool hash_reset_pending{};
    // Pending writes (stage A): the guest buffers that packets queued for stage B write, kept
    // until stage B has run them. Before stage A reads guest memory through a clean read (shader
    // resource tables) that one of them writes, it waits for that packet: the data it would read
    // is about to change, and only once the packet ran is the memory marked as GPU-written (read
    // back on access). Their pages go to the clean-read path of the SRT walkers, so the walkers'
    // plain loads do not miss them. Images written by shaders are not tracked
    struct PendingWrite {
        VAddr start;
        VAddr end;
        u64 position; ///< DrawPipe::Head() after the packet
    };
    static constexpr u64 PendingGranuleBits = 20;
    static constexpr u64 MaxPendingWrite = 64_MB; ///< larger writes count as unbounded
    void TrackPendingWrites(const Pipeline* pipeline, u64 position);
    void PrunePendingWrites();
    void WaitForPendingWrites(VAddr addr, u64 size);
    u32 pipe_extra{};                        ///< PINK_PIPE_EXTRA
    bool track_pending_writes{};             ///< tracking on (mode 7, "wait" or "ascwait")
    bool pending_for_srt{};                  ///< the tracking also guards SRT clean reads
    bool direct_draws{};                     ///< PINK_DIRECT_DRAWS
    u64 direct_draws_done{};
    u64 direct_draws_kept{};
    bool ordered_waits_unchecked{};          ///< PINK_PIPE_TRACK=0: skip without tracking
    std::deque<PendingWrite> pending_writes; ///< by position
    std::unique_ptr<u32[]> pending_granules; ///< pending writes per 1 MiB of the address space
    u64 pending_unbounded_position{};        ///< after the last packet with an unbounded write
    std::unordered_set<u64> srt_marked_ranges; ///< ranges already sent to MarkSrtCleanPages
    u64 pipe_waits_skipped{};
    u64 pending_writes_tracked{};
    u64 pending_writes_unbounded{};
    std::unique_ptr<DrawPreparation> draw_prep; ///< stops its helpers before the caches go
    std::unique_ptr<DrawPipe> draw_pipe; ///< last member: its thread stops before the rest goes
};

} // namespace Vulkan
