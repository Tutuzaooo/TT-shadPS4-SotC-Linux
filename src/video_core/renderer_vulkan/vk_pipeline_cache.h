// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <utility>
#include <variant>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <tsl/robin_map.h>
#include "common/assert.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
union Regs;
struct ShaderProgram;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    /// Draw preparation (vk_draw_prep.h): a copy of `info` as compiled, for helper threads, which
    /// must not touch `info` (the command processor thread refreshes it for every draw). Set under
    /// PipelineCache::programs_mutex.
    std::unique_ptr<Shader::Info> info_template;
    ModuleList modules{};
    /// Keep the indices of occupied modules slots in ascending order so searches skip the empty
    /// slots
    /// Stored permutation indices can leave gaps, and checking those gaps was most of the work in
    /// the old search
    /// AllocateShaderBinaryIndex keeps the stored indices stable, so we must keep the same order
    /// without renumbering them
    boost::container::small_vector<u32, MaxPermutations> valid_permuts{};

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        valid_permuts.push_back(static_cast<u32>(modules.size()));
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
        const auto it = std::ranges::lower_bound(valid_permuts, static_cast<u32>(perm_idx));
        if (it == valid_permuts.end() || *it != perm_idx) {
            valid_permuts.insert(it, static_cast<u32>(perm_idx));
        }
    }

    /// Find the first occupied permutation with a matching specialization, keeping the same index
    /// order as a full modules search
    /// Skipping an empty slot should never change which matching permutation is selected
    std::optional<size_t> FindPermut(const Shader::StageSpecialization& spec) const {
        std::optional<size_t> found{};
        for (const u32 idx : valid_permuts) {
            if (modules[idx].spec == spec) {
                found = idx;
                break;
            }
        }
        DEBUG_ASSERT(found.value_or(modules.size()) ==
                     static_cast<size_t>(std::distance(
                         modules.begin(), std::ranges::find(modules, spec, &Module::spec))));
        return found;
    }
};

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;

    bool operator==(const DrawIndirectParams&) const = default;
};

struct PrepWorker;
struct PreparedDraw;

/// The state one pipeline selection works on. The command processor thread selects with the
/// pipeline cache's own members; a draw preparation helper (vk_draw_prep.h) with its own copies
/// and its replayed registers, through the same code.
struct PipelineSelection {
    const AmdGpu::Regs& regs;
    std::array<Shader::RuntimeInfo, MaxShaderStages>& runtime_infos;
    std::array<const Shader::Info*, MaxShaderStages>& infos;
    std::array<vk::ShaderModule, MaxShaderStages>& modules;
    Shader::Gcn::FetchShaderData*& fetch_shader;
    GraphicsPipelineKey& graphics_key;
    const DrawIndirectParams& draw_indirect_params;
    PrepWorker* worker; ///< set: lookups only, nothing is compiled or created
};

/// A draw preparation helper's selection state and its own copies of the programs' Info
struct PrepWorker {
    struct Stage {
        const Program* program;
        u64 hash;
        Shader::HwStage hw_stage;
        const std::vector<u32>* flat;
    };
    const AmdGpu::Regs* regs{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    DrawIndirectParams draw_indirect_params{};
    tsl::robin_map<const Program*, std::unique_ptr<Shader::Info>> program_infos;
    boost::container::static_vector<Stage, MaxShaderStages> stages;
    Shader::StageSpecialization spec;
    bool failed = false;

    PipelineSelection Selection() {
        return {*regs,        runtime_infos,        infos, modules, fetch_shader,
                graphics_key, draw_indirect_params, this};
    }
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    /// `prepared`: the draw preparation helpers' result for this draw, used when it was made from
    /// the same registers (`state_hash`, vk_draw_prep.h) and the same guest memory
    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {},
                                                const PreparedDraw* prepared = nullptr,
                                                u64 state_hash = 0);

    /// Draw preparation helper: selects the pipeline of a draw from `worker.regs` without
    /// compiling or creating anything. False when that would be needed, or on a guest memory
    /// fault (the helper's recovery point).
    bool PrepareGraphicsPipeline(PrepWorker& worker);

    /// Draw preparation statistics (command processor thread)
    struct PrepStats {
        u64 used{};
        u64 pending{};
        u64 unavailable{};
        u64 state_mismatch{};
        u64 code_mismatch{};
        u64 flat_mismatch{};
        u64 no_pipeline{};
    };
    PrepStats prep_stats{};

    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(PipelineSelection& sel, Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

    /// Clean guest memory reader provided by the rasterizer
    std::function<bool(VAddr, void*, u64)> read_clean_memory;
    /// Whether BindTextures binds this single T# as null
    /// (Rasterizer::CheckBindableImage), set by the rasterizer
    std::function<bool(const AmdGpu::Image&, const Shader::ImageResource&)> is_null_image;

private:
    /// Cached shader metadata and the guest bytes used to find it
    struct CachedBinaryInfo {
        // Search ranges and their bytes in order
        boost::container::small_vector<std::pair<u64, u32>, 2> ranges;
        std::vector<u8> bytes;
        AmdGpu::BinaryInfo info;
    };

    /// Reuse shader parameters while the code bytes stay unchanged and return nullopt when no
    /// valid binary info is found, so the caller can skip the draw or dispatch safely
    template <typename Program>
    std::optional<Shader::ShaderParams> GetParamsCached(const Program& pgm);
    bool CachedBytesMatch(const u32* code, const CachedBinaryInfo& cached);

    bool RefreshGraphicsKey(PipelineSelection& sel);
    bool RefreshGraphicsStages(PipelineSelection& sel);
    bool RefreshComputeKey();

    /// The command processor thread's selection (the members below)
    PipelineSelection OwnSelection();
    Result GetProgramForWorker(PrepWorker& worker, Shader::HwStage hw_stage,
                               const Shader::ShaderParams& params,
                               Shader::Backend::Bindings& binding,
                               const Shader::RuntimeInfo& runtime_info);
    /// A helper's shader parameters: from what the command processor thread found at that address
    std::optional<Shader::ShaderParams> WorkerParams(const AmdGpu::ShaderProgram& pgm);
    void PublishParams(const u32* code, u64 hash, u32 length);
    const GraphicsPipeline* UsePrepared(const PreparedDraw& prepared, const DrawIndirectParams& params,
                                        u64 state_hash);

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding, bool store = true);
    const Shader::RuntimeInfo& BuildRuntimeInfo(PipelineSelection& sel, Shader::HwStage stage,
                                                Shader::SwStage l_stage);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    /// Draw preparation: the command processor thread is the only writer of `program_cache` and
    /// of the programs' permutation lists and takes this exclusively to change them; helpers read
    /// them under it shared
    std::shared_mutex programs_mutex;
    struct PublishedParams {
        u64 hash;
        u32 length;
    };
    tsl::robin_map<const u32*, PublishedParams> published_params;
    std::shared_mutex published_params_mutex;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start
    tsl::robin_map<const u32*, CachedBinaryInfo> binary_info_cache;
    std::vector<u8> code_scratch;

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
