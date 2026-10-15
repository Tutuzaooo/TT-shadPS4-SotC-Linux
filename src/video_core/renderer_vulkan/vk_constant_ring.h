// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <optional>

#include "common/assert.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

/// Adapted from bbport's vk_constant_ring.h, deadinside28/bloodborne_pc 3c57da4, GPL-2.0-or-later; stage A copies small read-only buffers
/// and stage B binds them; offsets increase monotonically, and regions are reused only after the GPU completes their last submission,
/// tracked by stage B stamps retired on stage A
class ConstantRing {
public:
    /// SotC copies about 23 MB per frame with 2-3 frames in flight
    static constexpr u64 Capacity = 128_MB;

    ConstantRing(const Instance& instance, Scheduler& scheduler_)
        : scheduler{scheduler_},
          buffer{instance, 0, Capacity, VideoCore::MemoryType::Stream, "constant ring"} {
        ASSERT(!buffer.mapped_data.empty());
    }

    [[nodiscard]] vk::Buffer Handle() const noexcept {
        return buffer.Handle();
    }

    /// Use device-local host-visible memory when available, otherwise host memory that the GPU reads over the bus
    [[nodiscard]] bool IsDeviceLocal() const noexcept {
        return buffer.is_device_local;
    }

    /// Before packet copies on stage A, retire known completed stamps once the queue is half full; Allocate alone only retires when out of
    /// space, which can let frequent submits fill the queue before the ring wraps; this check avoids a driver call
    void BeginPacket() {
        full = false;
        if (stamps_head.load(std::memory_order_relaxed) -
                stamps_tail.load(std::memory_order_relaxed) >=
            stamps.size() / 2) {
            Retire(false);
        }
    }

    /// Allocate aligned bytes on stage A or return nullopt while the GPU uses the region; after a refusal, leave the packet's remaining
    /// bindings to stage B to avoid repeated driver queries
    std::optional<u64> Allocate(u64 size, u64 alignment) {
        if (full) {
            return std::nullopt;
        }
        u64 at = (position + alignment - 1) & ~(alignment - 1);
        if (at % Capacity + size > Capacity) {
            at += Capacity - at % Capacity; // Wrap the allocation to the ring start
        }
        const auto in_use = [&] { return at + size > Capacity && at + size - Capacity > retired; };
        if (in_use()) {
            Retire(true);
            if (in_use()) {
                full = true;
                return std::nullopt;
            }
        }
        position = at + size;
        return at % Capacity;
    }

    [[nodiscard]] u8* Data(u64 offset) noexcept {
        return buffer.mapped_data.data() + offset;
    }

    /// Make stage A's completed writes visible to the GPU
    void Flush(u64 offset, u64 size) {
        if (!buffer.is_coherent) {
            buffer.Flush(offset, size);
        }
    }

    /// End offset of allocations in packets built so far
    [[nodiscard]] u64 Position() const noexcept {
        return position;
    }

    /// Stamp a recorded packet's end offset on stage B, merging ends within one submission tick; publish tick changes even without
    /// allocations so a ring filled in one tick can be reclaimed
    void Stamp(u64 end) {
        const u64 tick = scheduler.CurrentTick();
        if (end == pending_end && tick == pending_tick) {
            return;
        }
        if (tick != pending_tick && pending_end != last_published) {
            Publish(pending_end, pending_tick);
        }
        pending_end = end;
        pending_tick = tick;
    }

private:
    void Publish(u64 end, u64 tick) {
        const u64 at = stamps_head.load(std::memory_order_relaxed);
        if (at - stamps_tail.load(std::memory_order_acquire) >= stamps.size()) {
            return; // If the stamp queue is full, reclaim this region with a later stamp
        }
        stamps[at % stamps.size()] = {end, tick};
        stamps_head.store(at + 1, std::memory_order_release);
        last_published = end;
    }

    /// Reclaim regions whose submission tick has completed; ask_driver refreshes the known GPU tick at the first unfinished stamp
    void Retire(bool ask_driver) {
        const Semaphore& semaphore = *scheduler.GetWorkSemaphore();
        u64 at = stamps_tail.load(std::memory_order_relaxed);
        const u64 head = stamps_head.load(std::memory_order_acquire);
        while (at < head) {
            const StampEntry& stamp = stamps[at % stamps.size()];
            if (!(ask_driver ? scheduler.IsFree(stamp.tick) : semaphore.IsFree(stamp.tick))) {
                break;
            }
            retired = stamp.end;
            ++at;
        }
        stamps_tail.store(at, std::memory_order_release);
    }

    struct StampEntry {
        u64 end;
        u64 tick;
    };
    Scheduler& scheduler;
    VideoCore::Buffer buffer;
    u64 position = 0; ///< Owned by stage A
    u64 retired = 0;  ///< Stage A's end offset for completed GPU work
    bool full = false; ///< Stage A refused an allocation in this packet
    u64 pending_end = 0, pending_tick = 0, last_published = 0; ///< Owned by stage B
    std::array<StampEntry, 1024> stamps{};
    alignas(64) std::atomic<u64> stamps_head{0}; ///< Written by stage B
    alignas(64) std::atomic<u64> stamps_tail{0}; ///< Written by stage A
};

} // namespace Vulkan
