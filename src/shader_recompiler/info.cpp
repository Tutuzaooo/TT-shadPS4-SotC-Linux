// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/info.h"

namespace Shader {

void Info::RefreshSharpTables() {
    constexpr VAddr AddressMask = 0xFFFFFFFFFFFFULL;
    auto* memory = Core::Memory::Instance();
    for (const SharpTable& table : sharp_tables) {
        u32* dst = flattened_ud_buf.data() + table.flatbuf_off;
        const u64 size = u64{table.NumDwords()} * sizeof(u32);
        const VAddr ptr = (VAddr{flattened_ud_buf[table.ptr_lo]} |
                           VAddr{flattened_ud_buf[table.ptr_hi]} << 32) &
                          AddressMask;
        const VAddr src = ptr + u64{table.offset_dw} * sizeof(u32);
        // Through the backing, under the memory manager lock: a garbage table pointer may point at free
        // address space, where a direct read is a host access violation. The tables are written by the CPU,
        // so the backing is current
        if (ptr == 0 || !memory->TryReadBacking(src, dst, size)) {
            // Null T#s: the array binds null images, like the unflattened sharp did
            std::memset(dst, 0, size);
        }
    }
}

void Info::MissingSnapshot() const noexcept {
    static std::atomic<u32> reports{0};
    static const bool fatal = [] {
        const char* env = std::getenv("PINK_PIPE_CHECK");
        return env && env[0] == '1';
    }();
    if (reports.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_ERROR(Render, "Draw pipe: the recording thread read user data of shader {:#x} "
                          "(stage {}) that is not in the packet ({} snapshots)",
                  pgm_hash, static_cast<u32>(sw_stage), num_ud_snapshots);
    }
    ASSERT_MSG(!fatal, "Draw pipe: user data of shader {:#x} read without a snapshot", pgm_hash);
}

u32 DynamicTsharpArraySize() {
    return std::min(EmulatorSettings.GetDynamicTsharpArraySize(), MAX_TSHARP_ARRAY_SIZE);
}

} // namespace Shader
