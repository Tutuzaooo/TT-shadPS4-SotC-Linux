// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <exception>

#include "common/arch.h"
#include "common/assert.h"
#include "common/perf_stats.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_draw_pipe.h"

#ifdef ARCH_X86_64
#include <immintrin.h>
#endif

namespace Vulkan {

namespace {
/// Spin briefly between packets arriving microseconds apart, then sleep during longer gaps between frames
constexpr auto SpinTime = std::chrono::microseconds{200};

void Pause() {
#ifdef ARCH_X86_64
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

/// Spin first, then yield while the other stage finishes its work
template <typename Pred>
void SpinUntil(Pred&& done) {
    for (u32 spins = 0; !done(); ++spins) {
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}
} // Anonymous namespace

DrawPipe::DrawPipe(Handler handler_, void* context_) : handler{handler_}, context{context_} {
    ASSERT_MSG(active.load(std::memory_order_relaxed) == nullptr,
               "draw_pipe: only one draw pipe is supported");
    ring = std::make_unique_for_overwrite<u8[]>(Capacity);
    thread = std::jthread([this](std::stop_token stop) { Run(stop); });
    active.store(this, std::memory_order_release);
}

DrawPipe::~DrawPipe() {
    // Keep the pipe active until drained because stage B may still own the command stream; another thread can call this, so wait on
    // published packets rather than head
    SpinUntil([this] {
        return consumed.load(std::memory_order_acquire) ==
               published.load(std::memory_order_acquire);
    });
    active.store(nullptr, std::memory_order_release);
    thread.request_stop();
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_one();
}

u8* DrawPipe::Begin(u32 size) {
    size = Align(size + sizeof(Header));
    ASSERT(size <= Capacity / 4);
    u64 at = head;
    const u64 offset = at % Capacity;
    if (offset + size > Capacity) {
        // Skip the ring's remaining bytes; a zero-size header marks the wrap
        WaitForSpace(at, Capacity - offset);
        reinterpret_cast<Header*>(ring.get() + offset)->size = 0;
        at += Capacity - offset;
        head = at;
        published.store(at, std::memory_order_release);
    }
    WaitForSpace(at, size);
    pending_size = size;
    auto* header = reinterpret_cast<Header*>(ring.get() + at % Capacity);
    header->size = size;
    header->payload = size - static_cast<u32>(sizeof(Header));
    return reinterpret_cast<u8*>(header + 1);
}

void DrawPipe::Commit() {
    head += pending_size;
    published.store(head, std::memory_order_seq_cst);
    ++packets;
    Common::PerfStats::Add(Common::PerfStats::Id::PipePackets);
    if (sleeping.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

void DrawPipe::Drain() {
    if (Idle()) {
        return;
    }
    const Common::PerfStats::ScopedTimer timer{Common::PerfStats::Id::PipeDrains,
                                               Common::PerfStats::Id::PipeDrainNs};
    SpinUntil([this] { return Idle(); });
}

void DrawPipe::WaitPublished() {
    // Ring offsets increase monotonically; wrap markers advance both offsets to the start
    const u64 target = published.load(std::memory_order_acquire);
    SpinUntil([&] { return consumed.load(std::memory_order_acquire) >= target; });
}

void DrawPipe::WaitForSpace(u64 at, u64 size) {
    // A full ring puts stage B 16 MiB behind
    SpinUntil([&] { return at + size - consumed.load(std::memory_order_acquire) <= Capacity; });
}

void DrawPipe::Run(std::stop_token stop) {
    Common::SetCurrentThreadName("shadPS4:DrawRec");
    Common::PerfStats::MarkDrawRecordThread();
    on_stage_b = true;
    // Cache these values locally because the opaque handler call would otherwise reload members for every packet
    const Handler run_packet = handler;
    void* const run_context = context;
    const u8* const base = ring.get();
    u64 at = 0;
    while (true) {
        u64 available = published.load(std::memory_order_acquire);
        if (available == at) {
            const auto spin_until = std::chrono::steady_clock::now() + SpinTime;
            for (u32 spins = 1; available == at; ++spins) {
                if (stop.stop_requested()) {
                    return;
                }
                Pause();
                if (!(spins & 255) && std::chrono::steady_clock::now() >= spin_until) {
                    // Commit and sleeping use sequentially consistent operations so either this check sees the packet or Commit sees the
                    // sleeping flag
                    const u32 seen = wake.load(std::memory_order_seq_cst);
                    sleeping.store(true, std::memory_order_seq_cst);
                    if (published.load(std::memory_order_seq_cst) == at && !stop.stop_requested()) {
                        wake.wait(seen, std::memory_order_seq_cst);
                    }
                    sleeping.store(false, std::memory_order_relaxed);
                }
                available = published.load(std::memory_order_acquire);
            }
        }
        const auto* header = reinterpret_cast<const Header*>(base + at % Capacity);
        if (header->size == 0) {
            at += Capacity - at % Capacity;
            consumed.store(at, std::memory_order_release);
            continue;
        }
        {
            const Common::PerfStats::ScopedNs busy{Common::PerfStats::Id::PipeBusyNs};
            // Log task failures like command processor exceptions instead of silently reaching std::terminate
            try {
                run_packet(run_context, reinterpret_cast<const u8*>(header + 1), header->payload);
            } catch (const std::exception& e) {
                UNREACHABLE_MSG("draw_pipe: unhandled exception on the draw recording thread: {}",
                                e.what());
            }
        }
        at += header->size;
        // Only one thread writes this counter, so no atomic add is needed
        consumed_packets.store(consumed_packets.load(std::memory_order_relaxed) + 1,
                               std::memory_order_release);
        consumed.store(at, std::memory_order_release);
    }
}

} // namespace Vulkan
