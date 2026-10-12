// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Two-stage draw pipeline, after the one of the Bloodborne PC port (bbport, vk_draw_pipe.h).
//
// The GPU command processor thread (stage A) decodes PM4, keeps the register file and selects
// pipelines. For a draw or dispatch it writes a packet (the register blocks changed since the
// previous packet, the user data of the stages, the draw parameters) into this ring and goes on
// decoding. The draw recording thread (stage B) owns the caches, the barriers and the command
// recording while packets are in flight: it applies each packet to its own copy of the registers
// and runs the rest of the draw. Anything else stage A does on that state first waits for stage B
// to run dry (Drain) and then runs on stage A as before.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <x86intrin.h>
#ifdef __linux__
#include <unistd.h>
#endif

#include "common/assert.h"
#include "common/perf_stats.h"
#include "common/thread.h"
#include "common/types.h"

namespace Vulkan {

class DrawPipe {
public:
    /// Runs a packet on stage B.
    using Handler = void (*)(void* context, const u8* packet, u32 size);

    DrawPipe(Handler handler_, void* context_) : handler{handler_}, context{context_} {
        ring = std::make_unique<u8[]>(Capacity);
        instance = this;
        thread = std::jthread([this](std::stop_token stop) { Run(stop); });
        // The thread id is needed before the first packet (fault handling, command recorder)
        while (stage_b_tid.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }
    }

    ~DrawPipe() {
        instance = nullptr;
        thread.request_stop();
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }

    /// The command processor thread calls this once.
    static void MarkStageA() noexcept {
        on_stage_a = true;
    }

    /// Diagnostic, at entry points of the state stage B owns while packets are in flight (caches,
    /// scheduler, command recorder): stage A must have waited for stage B before it gets here.
    /// Aborts on the spot otherwise, so the crash report shows who came in and from where.
    static void CheckOwner(const char* where) {
        if (on_stage_a) [[unlikely]] {
            const DrawPipe* pipe = instance.load(std::memory_order_relaxed);
            if (pipe && !pipe->Idle()) {
                UNREACHABLE_MSG("Draw pipe: the command processor thread is in {} while draws "
                                "are queued for the recording thread",
                                where);
            }
        }
    }

    /// PINK_PIPE_CHECK=1: slower checks (packet contents stay intact while a packet runs, user
    /// data read without a snapshot aborts)
    [[nodiscard]] static bool Checks() noexcept {
        static const bool checks = [] {
            const char* env = std::getenv("PINK_PIPE_CHECK");
            return env && env[0] == '1';
        }();
        return checks;
    }

    /// True on the draw recording thread.
    [[nodiscard]] static bool OnStageB() noexcept {
        return on_stage_b;
    }

    [[nodiscard]] u32 StageBThreadId() const noexcept {
        return stage_b_tid.load(std::memory_order_acquire);
    }

    /// Stage A: space for a packet of `size` bytes, valid until Commit().
    u8* Begin(u32 size) {
        size = Align(size + sizeof(Header));
        ASSERT(size <= Capacity / 4);
        u64 at = head;
        const u64 offset = at % Capacity;
        if (offset + size > Capacity) {
            // Wrap: the rest of the ring is skipped (a header with zero size marks it).
            WaitForSpace(at, Capacity - offset);
            reinterpret_cast<Header*>(ring.get() + offset)->size = 0;
            at += Capacity - offset;
            head = at;
            published.store(at, std::memory_order_seq_cst);
            WakeIfSleeping();
        }
        WaitForSpace(at, size);
        pending_size = size;
        auto* header = reinterpret_cast<Header*>(ring.get() + at % Capacity);
        header->size = size;
        return reinterpret_cast<u8*>(header + 1);
    }

    /// Stage A: hands the packet from Begin() to stage B.
    void Commit(u32 payload_size) {
        reinterpret_cast<Header*>(ring.get() + head % Capacity)->payload = payload_size;
        head += pending_size;
        published.store(head, std::memory_order_seq_cst);
        ++packets;
        WakeIfSleeping();
    }

    /// Why stage A waited for stage B (statistics).
    enum Reason : u32 {
        ReasonRasterizer, ///< a rasterizer entry point other than a draw
        ReasonOpcode,     ///< a PM4 packet that is not handed to stage B
        ReasonCommands,   ///< work queued by other threads (readbacks, flips)
        ReasonConstRam,   ///< the constant engine writes guest memory
        ReasonSubmissionEnd,
        ReasonDispatch,
        ReasonFault, ///< a page fault of stage A
        ReasonLockstep,
        NumReasons,
    };
    static constexpr std::array<const char*, NumReasons> ReasonNames = {
        "rasterizer", "opcode", "commands", "constram", "submit_end", "dispatch", "fault",
        "lockstep",
    };
    std::array<u64, NumReasons> drains_by_reason{};
    std::array<u64, NumReasons> cycles_by_reason{};
    static constexpr u32 NumOpcodeSlots = 512;
    std::array<u64, NumOpcodeSlots> drains_by_opcode{};
    std::array<u64, NumOpcodeSlots> cycles_by_opcode{};

    /// Stage A: waits until stage B has run every committed packet.
    /// `detail` (ReasonOpcode): the PM4 opcode, plus 256 on a compute queue.
    void Drain(u32 reason, u32 detail = 0) {
        if (consumed.load(std::memory_order_acquire) == head) {
            return;
        }
        const u32 slot = reason < NumReasons ? reason : ReasonRasterizer;
        ++drains_by_reason[slot];
        const u32 opcode_slot = reason == ReasonOpcode ? detail % NumOpcodeSlots : 0;
        if (reason == ReasonOpcode) {
            ++drains_by_opcode[opcode_slot];
        }
        const u64 start = __rdtsc();
        for (u32 spins = 0; consumed.load(std::memory_order_acquire) != head; ++spins) {
            if (spins < 4096) {
                _mm_pause();
            } else {
                std::this_thread::yield();
            }
        }
        const u64 waited = __rdtsc() - start;
        cycles_by_reason[slot] += waited;
        if (reason == ReasonOpcode) {
            cycles_by_opcode[opcode_slot] += waited;
        }
    }

    [[nodiscard]] bool Idle() const noexcept {
        return consumed.load(std::memory_order_acquire) == head;
    }

    /// Stage A: the position after the last committed packet; Reached(position) once stage B
    /// has run everything up to it.
    [[nodiscard]] u64 Head() const noexcept {
        return head;
    }
    [[nodiscard]] bool Reached(u64 position) const noexcept {
        return consumed.load(std::memory_order_acquire) >= position;
    }

    /// Statistics: packets committed (stage A), cycles stage B spent running packets.
    u64 packets = 0;
    std::atomic<u64> busy_cycles{0};

private:
    struct Header {
        u32 size;    ///< bytes to the next packet (0: wrap to the ring start)
        u32 payload; ///< bytes written by the producer
    };
    static constexpr u64 Capacity = 16ull << 20;

    static constexpr u32 Align(u32 size) {
        return (size + 63) & ~63u;
    }

    void WakeIfSleeping() {
        if (sleeping.load(std::memory_order_seq_cst)) {
            wake.fetch_add(1, std::memory_order_seq_cst);
            wake.notify_one();
        }
    }

    void WaitForSpace(u64 at, u64 size) {
        while (at + size - consumed.load(std::memory_order_acquire) > Capacity) {
            _mm_pause();
        }
    }

    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:DrawRec");
        on_stage_b = true;
        Common::PerfStats::MarkGpuThread();
#ifdef __linux__
        stage_b_tid.store(static_cast<u32>(gettid()), std::memory_order_release);
#else
        stage_b_tid.store(1, std::memory_order_release);
#endif
        // Packets follow each other within microseconds while a frame is decoded: spin, and sleep
        // only after a longer pause (between frames). PINK_PIPE_SPIN_US overrides the 200 us.
        static const auto spin_time = std::chrono::microseconds([] {
            const char* env = std::getenv("PINK_PIPE_SPIN_US");
            return env ? std::max(0, std::atoi(env)) : 200;
        }());
        u64 at = 0;
        while (true) {
            u64 available = published.load(std::memory_order_acquire);
            if (available == at) {
                const auto spin_until = std::chrono::steady_clock::now() + spin_time;
                for (u32 spins = 1; available == at; ++spins) {
                    if (stop.stop_requested()) {
                        return;
                    }
                    _mm_pause();
                    if (!(spins & 255) && std::chrono::steady_clock::now() >= spin_until) {
                        const u32 seen = wake.load(std::memory_order_seq_cst);
                        sleeping.store(true, std::memory_order_seq_cst);
                        if (published.load(std::memory_order_seq_cst) == at &&
                            !stop.stop_requested()) {
                            wake.wait(seen, std::memory_order_seq_cst);
                        }
                        sleeping.store(false, std::memory_order_seq_cst);
                    }
                    available = published.load(std::memory_order_acquire);
                }
            }
            const auto* header = reinterpret_cast<const Header*>(ring.get() + at % Capacity);
            if (header->size == 0) {
                at += Capacity - at % Capacity;
                consumed.store(at, std::memory_order_release);
                continue;
            }
            const u64 start = __rdtsc();
            const auto* payload = reinterpret_cast<const u8*>(header + 1);
            const u32 payload_size = header->payload;
            const u32 packet_size = header->size;
            const bool checks = Checks();
            const u64 sum_before = checks ? Checksum(payload, payload_size) : 0;
            handler(context, payload, payload_size);
            if (checks) {
                ASSERT_MSG(Checksum(payload, payload_size) == sum_before &&
                               header->payload == payload_size && header->size == packet_size,
                           "Draw pipe: a packet changed while the recording thread ran it");
            }
            busy_cycles.fetch_add(__rdtsc() - start, std::memory_order_relaxed);
            at += header->size;
            consumed.store(at, std::memory_order_release);
        }
    }

    static u64 Checksum(const u8* data, u32 size) noexcept {
        u64 sum = 0xcbf29ce484222325ull;
        for (u32 i = 0; i + 8 <= size; i += 8) {
            u64 word;
            std::memcpy(&word, data + i, sizeof(word));
            sum = (sum ^ word) * 0x100000001b3ull;
        }
        return sum;
    }

    static inline thread_local bool on_stage_b = false;
    static inline thread_local bool on_stage_a = false;
    static inline std::atomic<DrawPipe*> instance{nullptr};
    Handler handler;
    void* context;
    std::unique_ptr<u8[]> ring;
    u64 head = 0; ///< stage A's write position
    u32 pending_size = 0;
    alignas(64) std::atomic<u64> published{0};
    alignas(64) std::atomic<u64> consumed{0};
    alignas(64) std::atomic<u32> wake{0};
    std::atomic<bool> sleeping{false};
    std::atomic<u32> stage_b_tid{0};
    std::jthread thread;
};

} // namespace Vulkan
