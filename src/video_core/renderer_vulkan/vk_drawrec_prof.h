// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// Where the draw recording thread spends its time per draw, with SHADPS4_PERF_STATS only
// (bash jogar-040.sh medir): one "DrawRec per draw" line with the draw pipe stats.

#include <array>
#include <atomic>
#include <x86intrin.h>
#include "common/perf_stats.h"
#include "common/types.h"

namespace Vulkan::DrawRecProf {

enum Section : u32 {
    Prep,
    Buffers,
    Textures,
    Begin,
    VertexIndex,
    Barriers,
    Push,
    State,
    Draw,
    Reset,
    Total,
    StreamCopy, // part of Buffers and VertexIndex
    SyncFlush,  // FlushSyncBatch, wherever it runs on this thread
    Count
};

inline constexpr std::array<const char*, Count> Names = {
    "prep",    "buffers", "textures", "begin", "vertex/index", "barriers", "push descriptors",
    "state",   "draw",    "reset",    "total", "stream copies", "sync flushes"};

inline std::array<std::atomic<u64>, Count> cycles{};
inline std::atomic<u64> draws{};

/// Buffer bindings by the path they took: their count and cycles (the whole loop iteration)
enum BufferPath : u32 {
    PathSpecial,
    PathFlatReused,
    PathReused,
    PathStream,
    PathStreamReused,
    PathArena,
    PathWritten,
    PathNull,
    NumPaths
};
inline constexpr std::array<const char*, NumPaths> PathNames = {
    "special", "user data reused", "reused", "stream", "stream reused", "arena", "written", "null"};
inline std::array<std::atomic<u64>, NumPaths> path_count{};
inline std::array<std::atomic<u64>, NumPaths> path_cycles{};
inline std::atomic<u64> stream_copies{};
inline std::atomic<u64> stream_bytes{};

inline bool On() {
    static const bool on = Common::PerfStats::Enabled();
    return on;
}

struct Scope {
    explicit Scope(u32 section_) : section{section_}, start{On() ? __rdtsc() : 0} {}
    ~Scope() {
        if (start != 0) {
            cycles[section].fetch_add(__rdtsc() - start, std::memory_order_relaxed);
        }
    }
    void Next(u32 next) {
        if (start != 0) {
            const u64 now = __rdtsc();
            cycles[section].fetch_add(now - start, std::memory_order_relaxed);
            start = now;
        }
        section = next;
    }
    u32 section;
    u64 start;
};

} // namespace Vulkan::DrawRecProf
