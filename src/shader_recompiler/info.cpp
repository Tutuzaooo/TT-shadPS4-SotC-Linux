// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/info.h"

namespace Shader {

void Info::FillSharpTables(u32* flat, bool* count_logged) const {
    constexpr VAddr AddressMask = 0xFFFFFFFFFFFFULL;
    auto* memory = Core::Memory::Instance();
    for (const SharpTable& table : sharp_tables) {
        u32* dst = flat + table.flatbuf_off;
        const u64 size = u64{table.NumDwords()} * sizeof(u32);
        u32 live = table.num_elements;
        if (table.count_loc != UNKNOWN_LOCATION) {
            // Entries past n are not T#s; in SotC's vs 0x9d6f9bf4 they look like a T# over the constant ring and caused a GPU hang
            const s32 count = static_cast<s32>(flat[table.count_loc]);
            live = count <= 0 ? 0u : std::min<u32>(count, table.num_elements);
            if (count > static_cast<s32>(table.num_elements) && count_logged && !*count_logged) {
                *count_logged = true;
                LOG_WARNING(
                    Render_Recompiler,
                    "Workaround dynamic_tsharp_array_size: shader {:#x} loops over {} T#s of a "
                    "table of {} (array too small, or a count read before the game wrote it)",
                    pgm_hash, count, table.num_elements);
            }
        }
        const u64 live_size = u64{live} * table.stride_dw * sizeof(u32);
        const VAddr ptr =
            (VAddr{flat[table.ptr_lo]} | VAddr{flat[table.ptr_hi]} << 32) & AddressMask;
        const VAddr src = ptr + u64{table.offset_dw} * sizeof(u32);
        // Read only live entries through the backing under the memory manager lock; invalid pointers can hit free address space, and
        // reading an unbacked tail would fail the whole read; CPU writes keep the backing current
        if (live_size != 0 && (ptr == 0 || !memory->TryReadBacking(src, dst, live_size))) {
            // Null T#s: the array binds null images, like the unflattened sharp did
            std::memset(dst, 0, live_size);
        }
        std::memset(reinterpret_cast<u8*>(dst) + live_size, 0, size - live_size);
    }
}

u32 DynamicTsharpArraySize() {
    return std::min(EmulatorSettings.GetDynamicTsharpArraySize(), MAX_TSHARP_ARRAY_SIZE);
}

} // namespace Shader
