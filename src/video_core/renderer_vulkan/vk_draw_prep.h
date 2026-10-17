// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Draw preparation on helper threads, after the one of the Bloodborne PC port (bbport,
// vk_draw_prep.h, docs/parallel_gpu.md).
//
// Pipeline selection was about 40% of the command processor thread (SotC, perf): for every draw
// it walks the resource tables of each shader stage, builds the stage specialization and looks for
// the matching permutation. Helper threads do that ahead of it:
//
// - The game's submit thread copies each graphics command buffer it submits.
// - A scanner thread replays the register writes of the copies in order, from the command
//   processor thread's own state, and cuts them into segments of up to 64 draws, each with the
//   hash of the register state at its start and the register blocks written inside it.
// - Workers (SCHED_IDLE: idle cores only) claim the segments nearest to the command processor
//   thread, reach their start state from the deltas and select the pipeline of each draw with the
//   same code the command processor thread uses (PipelineCache::PrepareGraphicsPipeline), without
//   creating anything.
// - The command processor thread uses a prepared draw only when the hash of its own register
//   state (kept up to date from the draw pipe's dirty blocks), the program hashes and the flattened
//   user data of every stage (everything read from guest memory) are the same; otherwise it selects
//   as before. A wrong guess costs a lookup, never a wrong pipeline.
//
// PINK_DRAW_PREP=1 turns it on (off by default), PINK_PREP_WORKERS=<n> sets the number of workers.

#pragma once

#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "common/types.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace AmdGpu {
union PM4Header;
}

namespace Vulkan {

/// Hash of a whole register file, kept up to date from the blocks written since the last update
struct RegStateHash {
    std::array<u64, AmdGpu::RegDirty::NumBlocks> blocks{};
    u64 value = 0;
    bool valid = false;

    void Full(const AmdGpu::Regs& regs);
    void Update(const AmdGpu::Regs& regs,
                const Common::BlockBits<AmdGpu::RegDirty::NumBlocks>& dirty);
};

struct PreparedStage {
    const Program* program;
    u64 hash;
    Shader::HwStage hw_stage;
    const u32* flat; ///< the flattened user data the worker computed (segment arena)
    u32 flat_size;
};

struct PreparedDraw {
    enum : u32 { Pending = 0, Ready = 1, Unavailable = 2 };
    std::atomic<u32> state{Pending};
    u64 state_hash{};
    DrawIndirectParams indirect_params{};
    GraphicsPipelineKey key{};
    u32 num_stages{};
    std::array<PreparedStage, MaxShaderStages> stages{};
};

class DrawPreparation {
public:
    DrawPreparation(PipelineCache& pipeline_cache, u32 workers);
    ~DrawPreparation();

    DrawPreparation(const DrawPreparation&) = delete;
    DrawPreparation& operator=(const DrawPreparation&) = delete;

    /// PINK_DRAW_PREP / PINK_PREP_WORKERS: the number of workers, 0 when off
    static u32 ConfiguredWorkers();

    struct Submission;
    /// Game submit thread: copies a graphics command buffer (outside the queue lock)
    std::shared_ptr<Submission> Build(std::span<const u32> commands);
    /// Game submit thread, under the queue lock: hands it over in submission order
    void Enqueue(u64 seq, std::shared_ptr<Submission> submission);

    /// Command processor thread: brackets a top-level graphics submission
    void BeginSubmission(u64 seq, const AmdGpu::Regs& regs, u64 state_hash);
    void EndSubmission();

    /// Command processor thread: the prepared state of the next draw packet of the submission,
    /// null when there is none
    const PreparedDraw* NextDraw();

    /// Command processor thread: one line every 10 s with SHADPS4_PERF_STATS
    void ReportStats(const PipelineCache::PrepStats& used);

    /// Register writes of a packet, as the command processor thread applies them (the ones that
    /// change `regs`). Returns false for packets it does not know to leave the registers alone.
    static void ApplyRegisterPacket(AmdGpu::Regs& regs, const AmdGpu::PM4Header* header,
                                    AmdGpu::RegDirty& dirty);

    struct Segment {
        u32 begin_word{}; ///< first packet, in words from the start of the commands
        u32 end_word{};
        u32 first_draw{};
        u32 num_draws{};
        u64 start_hash{};
        AmdGpu::RegDelta delta; ///< blocks written inside the segment, values at its end
        std::atomic<bool> claimed{false};
        std::vector<std::unique_ptr<u32[]>> arena; ///< flattened user data of its draws
    };

    struct Submission {
        u64 seq{};
        std::shared_ptr<const std::vector<u32>> commands;
        std::unique_ptr<PreparedDraw[]> draws;
        u32 num_draws{};
        std::vector<std::unique_ptr<Segment>> segments;
        u64 start_hash{}, end_hash{};
        bool opaque{}; ///< stopped at an indirect buffer: the end state is unknown
        std::atomic<bool> scanned{false};
        std::atomic<bool> gpu_done{false};
    };

private:
    void ScannerLoop(std::stop_token stop);
    void WorkerLoop(std::stop_token stop, u32 index);
    void Scan(Submission& submission, AmdGpu::Regs& regs, RegStateHash& hash);
    void PrepareSegment(Submission& submission, Segment& segment, PrepWorker& worker,
                        AmdGpu::Regs& regs);
    /// Under `mutex`: restart the helpers from the command processor thread's state
    void Rebase(u64 seq, const AmdGpu::Regs& regs, u64 state_hash);
    void Collect();

    PipelineCache& pipeline_cache;
    u32 worker_count;

    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<std::shared_ptr<Submission>> submissions; ///< by seq, from tail_seq
    /// The register state at the start of submission `tail_seq` (valid when `tail_valid`)
    std::unique_ptr<AmdGpu::Regs> tail_regs;
    u64 tail_seq{};
    u64 tail_hash{};
    bool tail_valid{};
    u64 epoch{}; ///< bumped by a rebase: the scanner and workers restart from the tail
    bool scanner_blocked{}; ///< the scanner stopped at an opaque submission
    std::chrono::steady_clock::time_point last_rebase{};

    // Command processor thread
    std::shared_ptr<Submission> current;
    u32 current_draw{};
    std::atomic<u64> gpu_seq{0};
    std::atomic<u32> gpu_draw{0};

    // Statistics
    std::atomic<u64> stat_prepared{0}, stat_failed{0}, stat_skipped{0}, stat_worker_ns{0};
    std::atomic<u64> stat_rebases{0}, stat_opaque{0}, stat_start_mismatch{0};
    u64 stat_no_submission{}, stat_draws_seen{};
    std::chrono::steady_clock::time_point stat_time{};

    std::jthread scanner;
    std::vector<std::jthread> workers;
};

} // namespace Vulkan
