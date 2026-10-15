// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "common/types.h"

namespace Vulkan {

/// Packet ring adapted from bbport's vk_draw_pipe.h, deadinside28/bloodborne_pc 3c57da4, GPL-2.0-or-later; stage A decodes PM4 and sends
/// draw state while stage B owns the caches, runtime and scheduler for recording; stage A drains before any other access to that state
class DrawPipe {
public:
    /// Execute a packet on stage B
    using Handler = void (*)(void* context, const u8* packet, u32 size);

    DrawPipe(Handler handler, void* context);
    ~DrawPipe();

    DrawPipe(const DrawPipe&) = delete;
    DrawPipe& operator=(const DrawPipe&) = delete;

    /// Check whether this is the draw recording thread
    [[nodiscard]] static bool OnStageB() noexcept {
        return on_stage_b;
    }
    /// Check whether MarkStageA identified this command processor thread
    [[nodiscard]] static bool OnStageA() noexcept {
        return on_stage_a;
    }
    static void MarkStageA() noexcept {
        on_stage_a = true;
    }
    /// Allow the command stream producer to switch to stage B or back to drained stage A; ring release/acquire operations order their
    /// writes; stage A recording before a drain is an error
    [[nodiscard]] static bool ProducerHandOffAllowed() noexcept {
        const DrawPipe* pipe = active.load(std::memory_order_acquire);
        return pipe && (on_stage_b || (on_stage_a && pipe->Idle()));
    }
    /// Check every stage A recording command for a drained pipe, regardless of the last producer; ProducerHandOffAllowed catches only
    /// producer changes
    [[nodiscard]] static bool StageAMayRecord() noexcept {
        const DrawPipe* pipe = active.load(std::memory_order_acquire);
        return !pipe || pipe->Idle();
    }

    /// Reserve size bytes aligned to 8 bytes on stage A; valid until Commit
    u8* Begin(u32 size);
    /// Publish the packet reserved by Begin to stage B
    void Commit();
    /// Wait on stage A until all committed packets have run
    void Drain();

    /// Number the next packet from 1 and check whether stage B has completed a given packet
    [[nodiscard]] u64 NextPacket() const noexcept {
        return packets + 1;
    }
    [[nodiscard]] bool ReachedPacket(u64 number) const noexcept {
        return consumed_packets.load(std::memory_order_acquire) >= number;
    }
    /// Read the completed packet count from any thread; stage B's current packet is not yet included
    [[nodiscard]] u64 ConsumedPackets() const noexcept {
        return consumed_packets.load(std::memory_order_acquire);
    }
    /// Outside stage A, wait for the packets published when called; guest threads use this before freeing memory
    void WaitPublished();
    /// Check on stage A whether all committed packets have run
    [[nodiscard]] bool Idle() const noexcept {
        return consumed.load(std::memory_order_acquire) == head;
    }

private:
    struct Header {
        u32 size;    ///< Bytes to the next packet; 0 wraps to the ring start
        u32 payload; ///< Number of bytes stage A has written
    };
    static constexpr u64 Capacity = 16ULL << 20;

    static constexpr u32 Align(u32 size) {
        return (size + 63) & ~63u;
    }

    void WaitForSpace(u64 at, u64 size);
    void Run(std::stop_token stop);

    static inline std::atomic<DrawPipe*> active{};
    static inline thread_local bool on_stage_a = false;
    static inline thread_local bool on_stage_b = false;
    /// Initialized once and read by both stages
    Handler handler;
    void* context;
    std::unique_ptr<u8[]> ring;
    /// Keep stage A's per-packet writes on a separate cache line from stage B's reads
    alignas(64) u64 head = 0; ///< Stage A's current write offset
    u32 pending_size = 0;
    u64 packets = 0;
    alignas(64) std::atomic<u64> published{0};
    alignas(64) std::atomic<u64> consumed{0};
    std::atomic<u64> consumed_packets{0}; ///< Number of packets completed by stage B
    alignas(64) std::atomic<u32> wake{0};
    std::atomic<bool> sleeping{false};
    /// Declare last so the thread joins before other members are destroyed
    std::jthread thread;
};

} // namespace Vulkan
