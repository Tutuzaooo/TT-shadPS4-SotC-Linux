// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/slot_vector.h"

namespace {

using Common::SlotId;

constexpr u32 ManyEntries = 5000;

std::string Name(u32 i) {
    // Force a heap allocation so copying a moved or freed entry exposes the problem
    return "entry number " + std::to_string(i) + " with a heap allocated name";
}

struct Counted {
    explicit Counted(int& live_) : live{&live_} {
        ++*live;
    }
    ~Counted() {
        --*live;
    }
    Counted(const Counted&) = delete;
    Counted& operator=(const Counted&) = delete;

    int* live;
};

TEST(SlotVector, EntriesKeepTheirAddressWhileTheVectorGrows) {
    Common::SlotVector<std::string> vector;
    const SlotId first = vector.insert(Name(0));
    const std::string* const address = &vector[first];
    std::vector<std::pair<SlotId, const std::string*>> entries;
    for (u32 i = 1; i < ManyEntries; ++i) {
        const SlotId id = vector.insert(Name(i));
        entries.emplace_back(id, &vector[id]);
    }
    EXPECT_EQ(&vector[first], address);
    EXPECT_EQ(*address, Name(0));
    for (u32 i = 0; i < entries.size(); ++i) {
        EXPECT_EQ(&vector[entries[i].first], entries[i].second);
        EXPECT_EQ(vector[entries[i].first], Name(i + 1));
    }
}

TEST(SlotVector, InsertCanCopyFromAnotherEntry) {
    // The texture cache builds images using info from another entry in the same vector
    Common::SlotVector<std::string> vector;
    std::vector<SlotId> ids{vector.insert(Name(0))};
    for (u32 i = 1; i < ManyEntries; ++i) {
        ids.push_back(vector.insert(vector[ids[i / 2]]));
    }
    for (u32 i = 1; i < ManyEntries; ++i) {
        EXPECT_EQ(vector[ids[i]], Name(0));
    }
}

TEST(SlotVector, IdsAreHandedOutInOrder) {
    Common::SlotVector<u32> vector;
    for (u32 i = 0; i < ManyEntries; ++i) {
        EXPECT_EQ(vector.insert(i).index, i);
    }
    EXPECT_EQ(vector.size(), ManyEntries);
}

TEST(SlotVector, ErasedSlotIsReusedFirst) {
    Common::SlotVector<u32> vector;
    for (u32 i = 0; i < ManyEntries; ++i) {
        vector.insert(i);
    }
    vector.erase(SlotId{10});
    vector.erase(SlotId{4000});
    EXPECT_EQ(vector.size(), ManyEntries - 2);
    EXPECT_FALSE(vector.is_allocated(SlotId{10}));
    EXPECT_FALSE(vector.is_allocated(SlotId{4000}));
    // Reuse the most recently erased id first
    EXPECT_EQ(vector.insert(1u).index, 4000u);
    EXPECT_EQ(vector.insert(2u).index, 10u);
    EXPECT_EQ(vector.insert(3u).index, ManyEntries);
    EXPECT_EQ(vector[SlotId{4000}], 1u);
    EXPECT_EQ(vector[SlotId{10}], 2u);
    EXPECT_EQ(vector.size(), ManyEntries + 1);
}

TEST(SlotVector, IsAllocatedRejectsIdsOutsideTheVector) {
    Common::SlotVector<u32> vector;
    EXPECT_FALSE(vector.is_allocated(SlotId{0}));
    EXPECT_FALSE(vector.is_allocated(SlotId{}));
    const SlotId id = vector.insert(7u);
    EXPECT_TRUE(vector.is_allocated(id));
    EXPECT_FALSE(vector.is_allocated(SlotId{id.index + 1}));
    EXPECT_FALSE(vector.is_allocated(SlotId{1u << 30}));
    EXPECT_FALSE(vector.is_allocated(SlotId{}));
}

TEST(SlotVector, IterationVisitsLiveEntriesInIdOrder) {
    Common::SlotVector<u32> vector;
    for (u32 i = 0; i < ManyEntries; ++i) {
        vector.insert(i);
    }
    std::vector<u32> expected;
    for (u32 i = 0; i < ManyEntries; ++i) {
        if (i % 3 == 0 || i == 1023 || i == 1024) {
            vector.erase(SlotId{i});
        } else {
            expected.push_back(i);
        }
    }
    std::vector<u32> seen;
    for (const u32 value : vector) {
        seen.push_back(value);
    }
    EXPECT_EQ(seen, expected);
    EXPECT_EQ(vector.size(), expected.size());
}

TEST(SlotVector, EraseAndDestructorDestroyEntries) {
    int live = 0;
    {
        Common::SlotVector<Counted> vector;
        std::vector<SlotId> ids;
        for (u32 i = 0; i < ManyEntries; ++i) {
            ids.push_back(vector.insert(live));
        }
        EXPECT_EQ(live, static_cast<int>(ManyEntries));
        for (u32 i = 0; i < ManyEntries; i += 2) {
            vector.erase(ids[i]);
        }
        EXPECT_EQ(live, static_cast<int>(ManyEntries / 2));
    }
    EXPECT_EQ(live, 0);
}

} // namespace
