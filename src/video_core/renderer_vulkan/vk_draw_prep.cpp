// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Draw preparation on helper threads (see vk_draw_prep.h)

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <xxhash.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "common/logging/log.h"
#include "common/perf_stats.h"
#include "common/thread.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderer_vulkan/vk_draw_prep.h"

namespace Vulkan {

using AmdGpu::PM4Header;
using AmdGpu::PM4ItOpcode;
using AmdGpu::RegDirty;
using AmdGpu::Regs;

namespace {

constexpr u32 SegmentDraws = 64;

u64 HashBlock(const Regs& regs, u32 block) {
    return XXH3_64bits_withSeed(regs.reg_array.data() + block * RegDirty::BlockWords,
                                RegDirty::BlockWords * sizeof(u32), block);
}

/// The draw packets the command processor thread hands to Rasterizer::Draw/DrawIndirect, with
/// the indirect draw parameters it passes to the pipeline selection
bool DrawPacketParams(const PM4Header* header, DrawIndirectParams& params) {
    params = {};
    switch (header->type3.opcode) {
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
        return true;
    case PM4ItOpcode::DrawIndirect: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawIndirect*>(header);
        params = {static_cast<u16>(p->base_vtx_loc), static_cast<u16>(p->start_inst_loc)};
        return true;
    }
    case PM4ItOpcode::DrawIndirectMulti: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawIndirectMulti*>(header);
        params = {static_cast<u16>(p->base_vtx_loc), static_cast<u16>(p->start_inst_loc)};
        return true;
    }
    case PM4ItOpcode::DrawIndexIndirect: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawIndexIndirect*>(header);
        params = {static_cast<u16>(p->base_vtx_loc), static_cast<u16>(p->start_inst_loc)};
        return true;
    }
    case PM4ItOpcode::DrawIndexIndirectMulti: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawIndexIndirectMulti*>(header);
        params = {static_cast<u16>(p->base_vtx_loc), static_cast<u16>(p->start_inst_loc)};
        return true;
    }
    case PM4ItOpcode::DrawIndexIndirectCountMulti: {
        const auto* p =
            reinterpret_cast<const AmdGpu::PM4CmdDrawIndexIndirectCountMulti*>(header);
        params = {static_cast<u16>(p->base_vtx_loc), static_cast<u16>(p->start_inst_loc)};
        return true;
    }
    default:
        return false;
    }
}

/// Walks the packets of a command buffer from `begin` to `end` (words). `func(header, at, words)`
/// returns false to stop. Returns false on a malformed buffer.
template <typename Func>
bool ForEachPacket(std::span<const u32> commands, u32 begin, u32 end, Func&& func) {
    for (u32 at = begin; at < end;) {
        const auto* header = reinterpret_cast<const PM4Header*>(commands.data() + at);
        if (header->type == 2) {
            ++at;
            continue;
        }
        if (header->type != 3) {
            return false;
        }
        const u32 words = header->type3.NumWords() + 1;
        if (at + words > end) {
            return false;
        }
        if (!func(header, at, words)) {
            return true;
        }
        at += words;
    }
    return true;
}

/// Flattened user data of a segment's draws: chunks never move, so the command processor thread
/// can read a prepared draw while the worker goes on
struct FlatArena {
    static constexpr size_t ChunkWords = 16 * 1024;
    std::vector<std::unique_ptr<u32[]>> chunks;
    size_t used = ChunkWords;

    const u32* Append(const std::vector<u32>& data) {
        const size_t words = data.size();
        if (chunks.empty() || words > ChunkWords - std::min(used, ChunkWords)) {
            chunks.push_back(std::make_unique<u32[]>(std::max(ChunkWords, words)));
            used = 0;
        }
        u32* dst = chunks.back().get() + used;
        if (words) {
            std::memcpy(dst, data.data(), words * sizeof(u32));
        }
        used += words;
        return dst;
    }
};

void MakeBackground() {
#ifdef __linux__
    sched_param param{};
    pthread_setschedparam(pthread_self(), SCHED_IDLE, &param);
#endif
}

} // Anonymous namespace

void RegStateHash::Full(const Regs& regs) {
    value = 0;
    for (u32 block = 0; block < RegDirty::NumBlocks; ++block) {
        blocks[block] = HashBlock(regs, block);
        value ^= blocks[block];
    }
    valid = true;
}

void RegStateHash::Update(const Regs& regs, const std::bitset<RegDirty::NumBlocks>& dirty) {
    for (size_t block = dirty._Find_first(); block < dirty.size();
         block = dirty._Find_next(block)) {
        value ^= blocks[block];
        blocks[block] = HashBlock(regs, static_cast<u32>(block));
        value ^= blocks[block];
    }
}

void DrawPreparation::ApplyRegisterPacket(Regs& regs, const PM4Header* header, RegDirty& dirty) {
    const u32 count = header->type3.NumWords();
    const auto* body = reinterpret_cast<const u32*>(header) + 1;
    const auto set_regs = [&](u32 base) {
        const auto* set_data = reinterpret_cast<const AmdGpu::PM4CmdSetData*>(header);
        if (count == 0) {
            return;
        }
        const u32 first = base + set_data->reg_offset;
        const u32 num = std::min(count - 1, first < Regs::NumRegs ? Regs::NumRegs - first : 0u);
        if (num == 0) {
            return;
        }
        std::memcpy(&regs.reg_array[first], body + 1, num * sizeof(u32));
        dirty.Mark(first, num);
    };
    switch (header->type3.opcode) {
    case PM4ItOpcode::ClearState:
        regs.SetDefaults();
        dirty.reset = true;
        break;
    case PM4ItOpcode::SetConfigReg:
        set_regs(Regs::ConfigRegWordOffset);
        break;
    case PM4ItOpcode::SetContextReg:
        set_regs(Regs::ContextRegWordOffset);
        break;
    case PM4ItOpcode::SetShReg: {
        // The compute program registers go to the queue's compute state, not to these
        const auto* set_data = reinterpret_cast<const AmdGpu::PM4CmdSetData*>(header);
        if (set_data->reg_offset >= 0x200 &&
            set_data->reg_offset <= (0x200 + sizeof(AmdGpu::ComputeProgram) / 4)) {
            break;
        }
        set_regs(Regs::ShRegWordOffset);
        break;
    }
    case PM4ItOpcode::SetUconfigReg:
        set_regs(Regs::UconfigRegWordOffset);
        break;
    case PM4ItOpcode::IndexType: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawIndexType*>(header);
        regs.index_buffer_type.raw = p->raw;
        dirty.MarkField(regs, regs.index_buffer_type);
        break;
    }
    case PM4ItOpcode::DrawIndex2: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawIndex2*>(header);
        regs.max_index_size = p->max_size;
        regs.index_base_address.base_addr_lo = p->index_base_lo;
        regs.index_base_address.base_addr_hi = p->index_base_hi;
        regs.num_indices = p->index_count;
        regs.draw_initiator = p->draw_initiator;
        dirty.MarkField(regs, regs.max_index_size);
        dirty.MarkField(regs, regs.index_base_address);
        dirty.MarkField(regs, regs.num_indices);
        dirty.MarkField(regs, regs.draw_initiator);
        break;
    }
    case PM4ItOpcode::DrawIndexOffset2: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawIndexOffset2*>(header);
        regs.max_index_size = p->max_size;
        regs.num_indices = p->index_count;
        regs.draw_initiator = p->draw_initiator;
        dirty.MarkField(regs, regs.max_index_size);
        dirty.MarkField(regs, regs.num_indices);
        dirty.MarkField(regs, regs.draw_initiator);
        break;
    }
    case PM4ItOpcode::DrawIndexAuto: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawIndexAuto*>(header);
        regs.num_indices = p->index_count;
        regs.draw_initiator = p->draw_initiator;
        dirty.MarkField(regs, regs.num_indices);
        dirty.MarkField(regs, regs.draw_initiator);
        break;
    }
    case PM4ItOpcode::NumInstances: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawNumInstances*>(header);
        regs.num_instances.num_instances = p->num_instances;
        dirty.MarkField(regs, regs.num_instances);
        break;
    }
    case PM4ItOpcode::IndexBase: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawIndexBase*>(header);
        regs.index_base_address.base_addr_lo = p->addr_lo;
        regs.index_base_address.base_addr_hi = p->addr_hi;
        dirty.MarkField(regs, regs.index_base_address);
        break;
    }
    case PM4ItOpcode::IndexBufferSize: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdDrawIndexBufferSize*>(header);
        regs.num_indices = p->num_indices;
        dirty.MarkField(regs, regs.num_indices);
        break;
    }
    case PM4ItOpcode::EventWrite: {
        const auto* p = reinterpret_cast<const AmdGpu::PM4CmdEventWrite*>(header);
        if (p->event_type.Value() == AmdGpu::EventType::SoVgtStreamoutFlush) {
            regs.cp_strmout_cntl.offset_update_done = 1;
            dirty.MarkField(regs, regs.cp_strmout_cntl);
        }
        break;
    }
    default:
        break;
    }
}

u32 DrawPreparation::ConfiguredWorkers() {
    // Off unless PINK_DRAW_PREP=1: in SotC most draws read resource tables that earlier
    // dispatches of the same frame write, so the helpers' early reads rarely match (2-16% of the
    // draws used a prepared pipeline, at no gain, 12/10/2026)
    if (const char* env = std::getenv("PINK_DRAW_PREP"); !env || env[0] != '1') {
        return 0;
    }
#ifndef __linux__
    return 0; // the helpers' recovery from guest memory faults is Linux only
#else
    if (const char* env = std::getenv("PINK_PREP_WORKERS")) {
        return static_cast<u32>(std::clamp(std::atoi(env), 0, 16));
    }
    // Half the hardware threads (6 on a 6-core/12-thread CPU), at most 8: they only take idle
    // cores
    return std::clamp<u32>(std::thread::hardware_concurrency() / 2, 1, 8);
#endif
}

DrawPreparation::DrawPreparation(PipelineCache& pipeline_cache_, u32 workers_)
    : pipeline_cache{pipeline_cache_}, worker_count{workers_} {
    tail_regs = std::make_unique<Regs>();
    stat_time = std::chrono::steady_clock::now();
    scanner = std::jthread([this](std::stop_token stop) { ScannerLoop(stop); });
    for (u32 i = 0; i < worker_count; ++i) {
        workers.emplace_back([this, i](std::stop_token stop) { WorkerLoop(stop, i); });
    }
    LOG_WARNING(Render_Vulkan,
                "Draw preparation: {} helper threads select pipelines ahead of the command "
                "processor (+1 scanner, {} hardware threads)",
                worker_count, std::thread::hardware_concurrency());
}

DrawPreparation::~DrawPreparation() {
    for (auto& worker : workers) {
        worker.request_stop();
    }
    scanner.request_stop();
    cv.notify_all();
    workers.clear();
    scanner = {};
}

std::shared_ptr<DrawPreparation::Submission> DrawPreparation::Build(
    std::span<const u32> commands) {
    auto submission = std::make_shared<Submission>();
    submission->commands =
        std::make_shared<const std::vector<u32>>(commands.begin(), commands.end());
    return submission;
}

void DrawPreparation::Enqueue(u64 seq, std::shared_ptr<Submission> submission) {
    if (!submission) {
        return;
    }
    submission->seq = seq;
    {
        std::scoped_lock lk{mutex};
        // Before the first submission the command processor began (no tail yet), or one it already
        // passed, the helpers have no state to start from
        if (!submissions.empty() && submissions.back()->seq + 1 != seq) {
            submissions.clear();
            tail_valid = false;
        }
        submissions.push_back(std::move(submission));
    }
    cv.notify_all();
}

void DrawPreparation::Rebase(u64 seq, const Regs& regs, u64 state_hash) {
    *tail_regs = regs;
    tail_seq = seq;
    tail_hash = state_hash;
    tail_valid = true;
    scanner_blocked = false;
    ++epoch;
    ++stat_rebases;
    last_rebase = std::chrono::steady_clock::now();
    // What was scanned from the old state is useless: fresh copies, scanned again
    std::deque<std::shared_ptr<Submission>> fresh;
    for (const auto& old : submissions) {
        if (old->seq < seq) {
            continue;
        }
        auto copy = std::make_shared<Submission>();
        copy->seq = old->seq;
        copy->commands = old->commands;
        fresh.push_back(std::move(copy));
    }
    submissions = std::move(fresh);
}

void DrawPreparation::BeginSubmission(u64 seq, const Regs& regs, u64 state_hash) {
    {
        std::scoped_lock lk{mutex};
        current.reset();
        current_draw = 0;
        gpu_seq.store(seq, std::memory_order_release);
        gpu_draw.store(0, std::memory_order_release);
        std::shared_ptr<Submission> found;
        if (!submissions.empty() && seq >= submissions.front()->seq &&
            seq - submissions.front()->seq < submissions.size()) {
            found = submissions[seq - submissions.front()->seq];
        }
        bool rebase = !tail_valid;
        if (found && found->scanned.load(std::memory_order_acquire)) {
            if (found->start_hash == state_hash) {
                current = found;
            } else {
                // Something the scanner does not replay changed the registers (an indirect
                // buffer, a compute queue): restart from the real state
                ++stat_start_mismatch;
                rebase = true;
            }
        } else if (scanner_blocked) {
            rebase = true;
        }
        // A rebase copies the whole register file and the helpers start over: not more than
        // once per 2 ms
        const auto now = std::chrono::steady_clock::now();
        if (rebase && (last_rebase == std::chrono::steady_clock::time_point{} ||
                       now - last_rebase >= std::chrono::milliseconds{2})) {
            Rebase(seq, regs, state_hash);
        }
    }
    cv.notify_all();
}

void DrawPreparation::EndSubmission() {
    current.reset();
    {
        std::scoped_lock lk{mutex};
        const u64 seq = gpu_seq.load(std::memory_order_relaxed);
        if (!submissions.empty() && seq >= submissions.front()->seq &&
            seq - submissions.front()->seq < submissions.size()) {
            submissions[seq - submissions.front()->seq]->gpu_done.store(
                true, std::memory_order_release);
        }
        Collect();
        gpu_seq.store(seq + 1, std::memory_order_release);
        gpu_draw.store(0, std::memory_order_release);
    }
    cv.notify_all();
}

void DrawPreparation::Collect() {
    while (!submissions.empty() && submissions.front()->gpu_done.load(std::memory_order_acquire)) {
        const auto& front = submissions.front();
        if (tail_valid && front->seq == tail_seq) {
            if (front->scanned.load(std::memory_order_acquire) && !front->opaque) {
                // Its deltas move the tail to the start of the next submission
                for (const auto& segment : front->segments) {
                    segment->delta.Apply(*tail_regs);
                }
                tail_hash = front->end_hash;
                ++tail_seq;
            } else {
                tail_valid = false;
            }
        }
        submissions.pop_front();
    }
}

const PreparedDraw* DrawPreparation::NextDraw() {
    ++stat_draws_seen;
    if (!current) {
        ++stat_no_submission;
        return nullptr;
    }
    const u32 index = current_draw++;
    gpu_draw.store(current_draw, std::memory_order_relaxed);
    if (index >= current->num_draws) {
        return nullptr;
    }
    return &current->draws[index];
}

void DrawPreparation::Scan(Submission& submission, Regs& regs, RegStateHash& hash) {
    const std::span<const u32> commands{*submission.commands};
    RegDirty dirty;
    u32 num_draws = 0;
    auto segment = std::make_unique<Segment>();
    segment->start_hash = hash.value;
    const auto close = [&](u32 end_word) {
        segment->end_word = end_word;
        segment->delta.Capture(regs, dirty);
        if (dirty.reset) {
            hash.Full(regs);
        } else {
            hash.Update(regs, dirty.blocks);
        }
        dirty.Clear();
        const u32 next_draw = segment->first_draw + segment->num_draws;
        submission.segments.push_back(std::move(segment));
        segment = std::make_unique<Segment>();
        segment->begin_word = end_word;
        segment->first_draw = next_draw;
        segment->start_hash = hash.value;
    };
    u32 last_end = 0;
    const bool complete = ForEachPacket(
        commands, 0, static_cast<u32>(commands.size()),
        [&](const PM4Header* header, u32 at, u32 words) {
            if (header->type3.opcode == PM4ItOpcode::IndirectBuffer) {
                // Its commands (and register writes) are elsewhere in guest memory: the state
                // after it is unknown
                submission.opaque = true;
                return false;
            }
            ApplyRegisterPacket(regs, header, dirty);
            last_end = at + words;
            DrawIndirectParams params;
            if (DrawPacketParams(header, params)) {
                ++num_draws;
                if (++segment->num_draws == SegmentDraws) {
                    close(at + words);
                }
            }
            return true;
        });
    if (!complete) {
        submission.opaque = true;
    }
    if (submission.opaque) {
        // Up to the indirect buffer: its draws can still be prepared
        if (segment->num_draws > 0) {
            close(last_end);
        }
    } else {
        close(static_cast<u32>(commands.size()));
    }
    submission.num_draws = num_draws;
    submission.draws = std::make_unique<PreparedDraw[]>(num_draws);
    submission.end_hash = hash.value;
}

void DrawPreparation::ScannerLoop(std::stop_token stop) {
    Common::SetCurrentThreadName("shadPS4:PrepScn");
    MakeBackground();
    auto regs = std::make_unique<Regs>();
    RegStateHash hash;
    u64 next_seq = 0;
    u64 seen_epoch = ~0ULL;
    while (!stop.stop_requested()) {
        std::shared_ptr<Submission> submission;
        {
            std::unique_lock lk{mutex};
            cv.wait(lk, stop, [&] {
                if (!tail_valid) {
                    return false;
                }
                if (epoch != seen_epoch) {
                    return true;
                }
                if (scanner_blocked || submissions.empty()) {
                    return false;
                }
                const u64 front = submissions.front()->seq;
                return next_seq >= front && next_seq - front < submissions.size();
            });
            if (stop.stop_requested()) {
                return;
            }
            if (epoch != seen_epoch) {
                *regs = *tail_regs;
                next_seq = tail_seq;
                seen_epoch = epoch;
                lk.unlock();
                hash.Full(*regs);
                continue;
            }
            submission = submissions[next_seq - submissions.front()->seq];
        }
        submission->start_hash = hash.value;
        Scan(*submission, *regs, hash);
        {
            std::scoped_lock lk{mutex};
            if (epoch == seen_epoch) {
                submission->scanned.store(true, std::memory_order_release);
                if (submission->opaque) {
                    scanner_blocked = true;
                    ++stat_opaque;
                }
                ++next_seq;
            }
        }
        cv.notify_all();
    }
}

void DrawPreparation::PrepareSegment(Submission& submission, Segment& segment, PrepWorker& worker,
                                     Regs& regs) {
    const auto start = std::chrono::steady_clock::now();
    RegStateHash hash;
    hash.Full(regs);
    if (hash.value != segment.start_hash) {
        stat_skipped.fetch_add(segment.num_draws, std::memory_order_relaxed);
        return;
    }
    FlatArena arena;
    RegDirty dirty;
    u32 draw = segment.first_draw;
    const std::span<const u32> commands{*submission.commands};
    worker.regs = &regs;
    ForEachPacket(commands, segment.begin_word, segment.end_word,
                  [&](const PM4Header* header, u32, u32) {
                      ApplyRegisterPacket(regs, header, dirty);
                      DrawIndirectParams params;
                      if (!DrawPacketParams(header, params)) {
                          return true;
                      }
                      if (dirty.reset) {
                          hash.Full(regs);
                      } else {
                          hash.Update(regs, dirty.blocks);
                      }
                      dirty.Clear();
                      const u32 index = draw++;
                      // Draws the command processor thread already passed are not worth it
                      const u64 gpu = gpu_seq.load(std::memory_order_acquire);
                      if (submission.seq < gpu ||
                          (submission.seq == gpu &&
                           index < gpu_draw.load(std::memory_order_relaxed))) {
                          stat_skipped.fetch_add(1, std::memory_order_relaxed);
                          return true;
                      }
                      auto& out = submission.draws[index];
                      worker.draw_indirect_params = params;
                      if (!pipeline_cache.PrepareGraphicsPipeline(worker)) {
                          stat_failed.fetch_add(1, std::memory_order_relaxed);
                          out.state.store(PreparedDraw::Unavailable, std::memory_order_release);
                          return true;
                      }
                      out.state_hash = hash.value;
                      out.indirect_params = params;
                      out.key = worker.graphics_key;
                      out.num_stages = static_cast<u32>(worker.stages.size());
                      for (u32 i = 0; i < out.num_stages; ++i) {
                          const auto& stage = worker.stages[i];
                          out.stages[i] = {
                              .program = stage.program,
                              .hash = stage.hash,
                              .hw_stage = stage.hw_stage,
                              .flat = arena.Append(*stage.flat),
                              .flat_size = static_cast<u32>(stage.flat->size()),
                          };
                      }
                      stat_prepared.fetch_add(1, std::memory_order_relaxed);
                      out.state.store(PreparedDraw::Ready, std::memory_order_release);
                      return true;
                  });
    segment.arena = std::move(arena.chunks);
    stat_worker_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - start)
                                 .count(),
                             std::memory_order_relaxed);
}

void DrawPreparation::WorkerLoop(std::stop_token stop, u32 index) {
    Common::SetCurrentThreadName(("shadPS4:Prep" + std::to_string(index)).c_str());
    MakeBackground();
    auto regs = std::make_unique<Regs>();
    auto worker = std::make_unique<PrepWorker>();
    // `regs` holds the state at the start of segment (pos_seq, pos_segment) of epoch pos_epoch
    u64 pos_seq = 0;
    size_t pos_segment = 0;
    u64 pos_epoch = ~0ULL;
    std::vector<std::shared_ptr<Submission>> keep;
    std::vector<const AmdGpu::RegDelta*> path;

    while (!stop.stop_requested()) {
        std::shared_ptr<Submission> target;
        Segment* segment = nullptr;
        size_t segment_index = 0;
        {
            std::unique_lock lk{mutex};
            // The nearest scanned segment the command processor thread has not reached and no
            // worker claimed
            const auto claimable = [&] {
                if (!tail_valid) {
                    return false;
                }
                const u64 gs = gpu_seq.load(std::memory_order_acquire);
                const u32 gd = gpu_draw.load(std::memory_order_relaxed);
                for (const auto& submission : submissions) {
                    if (submission->seq < gs) {
                        continue;
                    }
                    if (!submission->scanned.load(std::memory_order_acquire)) {
                        return false;
                    }
                    for (size_t i = 0; i < submission->segments.size(); ++i) {
                        auto& candidate = *submission->segments[i];
                        if (candidate.claimed.load(std::memory_order_relaxed) ||
                            candidate.num_draws == 0) {
                            continue;
                        }
                        if (submission->seq == gs &&
                            candidate.first_draw + candidate.num_draws <= gd) {
                            continue;
                        }
                        target = submission;
                        segment = &candidate;
                        segment_index = i;
                        return true;
                    }
                    if (submission->opaque) {
                        return false;
                    }
                }
                return false;
            };
            cv.wait(lk, stop, claimable);
            if (stop.stop_requested() || !target) {
                return;
            }
            segment->claimed.store(true, std::memory_order_relaxed);
            // The register blocks written between this worker's state and the segment's start
            const u64 front = submissions.front()->seq;
            const bool behind_tail =
                pos_epoch != epoch || pos_seq < tail_seq || pos_seq < front ||
                pos_seq > target->seq ||
                (pos_seq == target->seq && pos_segment > segment_index);
            if (behind_tail) {
                *regs = *tail_regs;
                pos_seq = tail_seq;
                pos_segment = 0;
                pos_epoch = epoch;
            }
            keep.clear();
            path.clear();
            for (u64 seq = pos_seq; seq <= target->seq; ++seq) {
                const auto& step = submissions[seq - front];
                keep.push_back(step);
                const size_t first = seq == pos_seq ? pos_segment : 0;
                const size_t last = seq == target->seq ? segment_index : step->segments.size();
                for (size_t i = first; i < last; ++i) {
                    path.push_back(&step->segments[i]->delta);
                }
            }
        }
        for (const auto* delta : path) {
            delta->Apply(*regs);
        }
        PrepareSegment(*target, *segment, *worker, *regs);
        // Now at the start of the next segment
        if (segment_index + 1 < target->segments.size()) {
            pos_seq = target->seq;
            pos_segment = segment_index + 1;
        } else {
            pos_seq = target->seq + 1;
            pos_segment = 0;
        }
        if (target->opaque && segment_index + 1 >= target->segments.size()) {
            pos_epoch = ~0ULL; // the state after an opaque submission is unknown
        }
    }
}

void DrawPreparation::ReportStats(const PipelineCache::PrepStats& used) {
    if (!Common::PerfStats::Enabled()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - stat_time < std::chrono::seconds{10}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - stat_time).count();
    stat_time = now;
    static PipelineCache::PrepStats last{};
    const auto delta = [&](u64 PipelineCache::PrepStats::* field) {
        return used.*field - last.*field;
    };
    const u64 draws = stat_draws_seen;
    LOG_WARNING(Render_Vulkan,
                "Draw preparation: {} of {} draws used a prepared pipeline ({:.0f}%); not used: "
                "{} not ready, {} unavailable, {} other registers, {} other code, {} other memory, "
                "{} no pipeline, {} outside a scanned submission; helpers: {} prepared, {} failed, "
                "{} skipped, {:.0f}% of one core; scanner: {} rebases, {} start mismatches, {} "
                "submissions with an indirect buffer",
                delta(&PipelineCache::PrepStats::used), draws,
                draws ? 100.0 * delta(&PipelineCache::PrepStats::used) / draws : 0.0,
                delta(&PipelineCache::PrepStats::pending),
                delta(&PipelineCache::PrepStats::unavailable),
                delta(&PipelineCache::PrepStats::state_mismatch),
                delta(&PipelineCache::PrepStats::code_mismatch),
                delta(&PipelineCache::PrepStats::flat_mismatch),
                delta(&PipelineCache::PrepStats::no_pipeline), stat_no_submission,
                stat_prepared.exchange(0), stat_failed.exchange(0), stat_skipped.exchange(0),
                stat_worker_ns.exchange(0) / (seconds * 1e7), stat_rebases.exchange(0),
                stat_start_mismatch.exchange(0), stat_opaque.exchange(0));
    last = used;
    stat_draws_seen = 0;
    stat_no_submission = 0;
}

} // namespace Vulkan
