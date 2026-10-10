#pragma once

#include "sd15/converter.hpp"
#include "tensor_transform.hpp"

#include <unordered_map>

namespace sd15::detail {

inline constexpr char kFormatName[] = "sd15-mnn-template";
inline constexpr char kOwnershipFile[] = ".sd15-converter.json";
inline constexpr char kTemplateFilename[] = "graph.bin";
inline constexpr std::size_t kMaxOutputFiles = 16;

struct SourceTensor {
    uint64_t offset;
    uint64_t elements;
    std::string dtype;
    std::vector<uint64_t> shape;
};
using SourceIndex = std::unordered_map<std::string, SourceTensor>;

enum class TaskKind { Literal, Zero, Float, Hqq };

// 위치와 설정만 소유한다. JSON·가중치·파일 핸들의 수명에 의존하지 않는다.
struct ConversionTask {
    TaskKind kind = TaskKind::Zero;
    uint64_t output_offset = 0;
    uint64_t size = 0;
    uint64_t source_offset = 0;
    uint64_t elements = 0;
    FloatStorage source_storage = FloatStorage::F32;
    FloatStorage target_storage = FloatStorage::F32;
    bool positive_zero = false;
    std::string tensor;
    uint64_t alpha_offset = 0;
    uint64_t group_count = 0;
    std::size_t group_elements = 0;
};

struct FilePlan {
    std::string name;
    uint64_t size = 0;
    std::vector<ConversionTask> tasks;
};

struct ConversionPlan {
    int manifest_version = 0;
    std::string mnn_version;
    std::string unet_quantization;
    uint64_t output_bytes = 0;
    bool has_hqq = false;
    std::vector<FilePlan> files;
};

bool safe_filename(const std::string& name);
// 실패 시 plan을 공개하지 않는다. JSON은 함수 반환 전에 해제된다.
bool build_conversion_plan(const std::vector<uint8_t>& bytes, uint64_t template_size,
                           const SourceIndex& source, ConversionPlan& plan, ConvertError& error);

}  // namespace sd15::detail
