// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_shader_hle.h"

namespace {

using Vulkan::ReplicateFillPlan;
using Vulkan::ReplicateFillShape;
using Vulkan::SizeReplicateFill;

constexpr VAddr Ctl = 0x200000000ULL;
constexpr VAddr Src = 0x200010000ULL;
constexpr VAddr Dst = 0x200020000ULL;

// Fill 100 elements from a 7-element source with no tighter limits
ReplicateFillShape Shape() {
    return {
        .ctl = Ctl,
        .src = Src,
        .dst = Dst,
        .src_size = 4096,
        .dst_size = 4096,
        .n = 100,
        .m = 7,
        .limit = 1000,
        .threads = 128,
    };
}

TEST(ReplicateFill, RepeatMatchesModulo) {
    for (u64 m = 1; m <= 40; ++m) {
        for (u64 count = 0; count <= 200; ++count) {
            const u64 src_count = std::min(m, count);
            std::vector<u32> data(count);
            for (u64 i = 0; i < src_count; ++i) {
                data[i] = 0x1000u + static_cast<u32>(i);
            }
            Vulkan::RepeatReplicateFill(data, src_count);
            for (u64 i = 0; i < count; ++i) {
                ASSERT_EQ(data[i], 0x1000u + static_cast<u32>(i % m))
                    << "m " << m << " count " << count << " i " << i;
            }
        }
    }
}

TEST(ReplicateFill, RepeatWithoutSourceLeavesData) {
    std::vector<u32> data{1, 2, 3};
    Vulkan::RepeatReplicateFill(data, 0);
    EXPECT_EQ(data, (std::vector<u32>{1, 2, 3}));
}

TEST(ReplicateFill, CountIsTheSmallestLimit) {
    const auto size = SizeReplicateFill(Shape());
    EXPECT_EQ(size.plan, ReplicateFillPlan::Fill);
    EXPECT_EQ(size.count, 100u);
    EXPECT_EQ(size.src_count, 7u);

    auto shape = Shape();
    shape.limit = 50;
    EXPECT_EQ(SizeReplicateFill(shape).count, 50u);
    shape = Shape();
    shape.threads = 64;
    EXPECT_EQ(SizeReplicateFill(shape).count, 64u);
    shape = Shape();
    shape.dst_size = 43; // Discard stores beyond the destination, including a partial dword
    EXPECT_EQ(SizeReplicateFill(shape).count, 10u);
    // Read the source once when m exceeds the output count
    shape = Shape();
    shape.m = 500;
    const auto once = SizeReplicateFill(shape);
    EXPECT_EQ(once.plan, ReplicateFillPlan::Fill);
    EXPECT_EQ(once.src_count, 100u);
}

TEST(ReplicateFill, NothingToWriteIsEmpty) {
    for (const auto zero : {&ReplicateFillShape::n, &ReplicateFillShape::limit}) {
        auto shape = Shape();
        shape.*zero = 0;
        EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Empty);
    }
    auto shape = Shape();
    shape.threads = 0; // A zero dispatch dimension produces no work
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Empty);
    shape = Shape();
    shape.dst_size = 3;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Empty);
    // An empty fill does not read m or the source
    shape = Shape();
    shape.n = 0;
    shape.m = 0;
    shape.src_size = 0;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Empty);
}

TEST(ReplicateFill, UnalignedBuffersAreRefused) {
    for (const auto field :
         {&ReplicateFillShape::ctl, &ReplicateFillShape::src, &ReplicateFillShape::dst}) {
        for (const VAddr offset : {1, 2, 3}) {
            auto shape = Shape();
            shape.*field += offset;
            EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Refused);
            // Validate control addresses before using the count; the GPU may read them elsewhere
            shape.n = 0;
            EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Refused);
        }
    }
}

TEST(ReplicateFill, LargeFillsAreRefused) {
    auto shape = Shape();
    shape.n = 16384;
    shape.limit = 16384;
    shape.threads = 16384;
    shape.dst_size = 64 * 1024;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Fill);
    shape.n = shape.limit = 16385;
    shape.threads = 16385;
    shape.dst_size = 64 * 1024 + 4;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Refused);
}

TEST(ReplicateFill, DivisionByZeroIsRefused) {
    auto shape = Shape();
    shape.m = 0;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Refused);
}

TEST(ReplicateFill, ShortSourceIsRefused) {
    auto shape = Shape();
    shape.src_size = 7 * 4;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Fill);
    shape.src_size = 7 * 4 - 1;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Refused);
}

TEST(ReplicateFill, OverlapsWithTheDestinationAreRefused) {
    auto shape = Shape();
    // Make the 7-element source touch either end of the destination
    shape.src = Dst - 6 * 4;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Refused);
    shape.src = Dst + 99 * 4;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Refused);
    // Test ranges immediately before and after the destination
    shape.src = Dst - 7 * 4;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Fill);
    shape.src = Dst + 100 * 4;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Fill);

    shape = Shape();
    shape.ctl = Dst - 4;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Refused);
    shape.ctl = Dst + 99 * 4;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Refused);
    shape.ctl = Dst - 8;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Fill);
    shape.ctl = Dst + 100 * 4;
    EXPECT_EQ(SizeReplicateFill(shape).plan, ReplicateFillPlan::Fill);
}

} // namespace
