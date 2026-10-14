// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
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
        if (table.count_loc != 0) {
            // Entries past n are not T#s; in SotC's vs 0x9d6f9bf4 they look like a T# over the constant ring and caused a GPU hang
            const s32 count = static_cast<s32>(flattened_ud_buf[table.count_loc]);
            const u32 live = count <= 0 ? 0u : std::min<u32>(count, table.num_elements);
            std::memset(dst + live * table.stride_dw, 0,
                        u64{table.num_elements - live} * table.stride_dw * sizeof(u32));
        }
    }
}

u32 DynamicTsharpArraySize() {
    return std::min(EmulatorSettings.GetDynamicTsharpArraySize(), MAX_TSHARP_ARRAY_SIZE);
}

} // namespace Shader
