#include "conversion_plan.hpp"
#include "conversion_error.hpp"
#include "conversion_json.hpp"

#include <cstdio>

namespace sd15::detail {
namespace {

constexpr int kFloatVersion = 1;
constexpr int kHqqVersion = 2;
constexpr std::size_t kMaxSegments = 65536;
constexpr char kMnnVersion[] = "3.6.1";

struct HqqPair {
    const Json* weight = nullptr;
    const Json* alpha = nullptr;
    std::size_t task_index = 0;
};

bool resolve_source(const Json& segment, const SourceIndex& source, ConversionTask& task, ConvertError& error) {
    try {
        const auto found = source.find(task.tensor);
        if (found == source.end()) {
            return fail(error, ErrorCode::TensorMismatch, "Missing tensor: " + task.tensor, {}, task.tensor);
        }
        const auto& tensor = found->second;
        if ((tensor.dtype != "F16" && tensor.dtype != "F32") || Json(tensor.shape) != segment.at("source_shape") ||
            !segment.at("positive_zero").is_boolean()) {
            return fail(error, ErrorCode::TensorMismatch, "Invalid tensor shape/dtype: " + task.tensor, {},
                        task.tensor);
        }
        task.source_offset = tensor.offset;
        task.elements = tensor.elements;
        task.source_storage = tensor.dtype == "F16" ? FloatStorage::F16 : FloatStorage::F32;
        task.positive_zero = segment.at("positive_zero");
        if (task.kind == TaskKind::Hqq) {
            return true;
        }
        const std::string dtype = segment.at("dtype");
        if (dtype != "F16" && dtype != "F32") {
            return fail(error, ErrorCode::TensorMismatch, "Only F16/F32 weights are supported", {}, task.tensor);
        }
        task.target_storage = dtype == "F16" ? FloatStorage::F16 : FloatStorage::F32;
        const auto width = storage_width(task.target_storage);
        if (element_count(segment.at("shape")) != tensor.elements || task.size % width != 0 ||
            task.size / width != tensor.elements) {
            return fail(error, ErrorCode::TensorMismatch, "Tensor shape/dtype/byte count mismatch", {}, task.tensor);
        }
        return true;
    }
    catch (const std::bad_alloc&) {
        throw;
    }
    catch (const std::exception& exception) {
        return fail(error, ErrorCode::TensorMismatch, exception.what(), {}, task.tensor);
    }
}

bool collect_hqq(const Json& segment, ConversionTask& task, HqqPair& pair, FilePlan& file, ConvertError& error) {
    try {
        const auto& recipe = segment.at("quantization");
        const uint64_t area = unsigned_value(recipe.at("group_elements"));
        const uint64_t groups = unsigned_value(recipe.at("group_count"));
        if (recipe.at("algorithm") != "hqq" || recipe.at("bits") != kHqqBits ||
            recipe.at("iterations") != kHqqIterations || recipe.at("lp_norm").get<float>() != kHqqLpNorm ||
            recipe.at("beta").get<float>() != kHqqBeta || area == 0 || area > kMaxHqqGroupElements || groups == 0 ||
            task.elements % area != 0 || task.elements / area != groups) {
            return fail(error, ErrorCode::IncompatibleTemplate, "Unsupported HQQ recipe", {}, task.tensor);
        }
        task.group_elements = static_cast<std::size_t>(area);
        task.group_count = groups;
        const std::string field = segment.at("field");
        const std::string dtype = segment.at("dtype");
        if (field == "Weight" && dtype == "U8" && task.size == task.elements &&
            element_count(segment.at("shape")) == task.elements && pair.weight == nullptr) {
            pair.weight = &segment;
            pair.task_index = file.tasks.size();
            file.tasks.push_back(task);
            return true;
        }
        if (field == "Alpha" && dtype == "F32" && groups <= UINT64_MAX / kHqqAlphaBytes &&
            task.size == groups * kHqqAlphaBytes && segment.at("shape") == Json::array({groups, kHqqAlphaValues}) &&
            pair.alpha == nullptr) {
            pair.alpha = &segment;
            return true;
        }
        return fail(error, ErrorCode::IncompatibleTemplate, "Invalid HQQ output", {}, task.tensor);
    }
    catch (const std::bad_alloc&) {
        throw;
    }
    catch (const std::exception& exception) {
        return fail(error, ErrorCode::IncompatibleTemplate, exception.what(), {}, task.tensor);
    }
}

bool build_file(const Json& description, uint64_t template_size, int version, const SourceIndex& source, FilePlan& file,
                ConvertError& error) {
    file.name = description.at("name");
    file.size = unsigned_value(description.at("size"));
    if (!safe_filename(file.name) || file.size > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        return fail(error, ErrorCode::IncompatibleTemplate, "Invalid output filename or size: " + file.name);
    }
    const auto& segments = description.at("segments");
    if (!segments.is_array() || segments.empty() || segments.size() > kMaxSegments) {
        return fail(error, ErrorCode::IncompatibleTemplate, file.name + ": invalid segments");
    }
    uint64_t cursor = 0;
    std::unordered_map<std::string, HqqPair> pairs;
    for (const auto& segment : segments) {
        ConversionTask task;
        task.output_offset = unsigned_value(segment.at("offset"));
        task.size = unsigned_value(segment.at("size"));
        const std::string kind = segment.at("kind");
        if (task.output_offset != cursor || task.size == 0 || task.size > file.size - cursor) {
            return fail(error, ErrorCode::IncompatibleTemplate, file.name + ": segment overlap, gap, or size mismatch");
        }
        cursor += task.size;
        if (kind == "zero") {
            file.tasks.push_back(std::move(task));
            continue;
        }
        if (kind == "literal") {
            task.kind = TaskKind::Literal;
            task.source_offset = unsigned_value(segment.at("template_offset"));
            if (task.source_offset > template_size || task.size > template_size - task.source_offset) {
                return fail(error, ErrorCode::IncompatibleTemplate, file.name + ": literal outside template");
            }
            file.tasks.push_back(std::move(task));
            continue;
        }
        if (kind != "tensor" && kind != "quantized") {
            return fail(error, ErrorCode::IncompatibleTemplate, file.name + ": unknown segment kind");
        }
        task.kind = kind == "tensor" ? TaskKind::Float : TaskKind::Hqq;
        task.tensor = segment.at("key");
        if (!resolve_source(segment, source, task, error)) {
            return false;
        }
        if (task.kind == TaskKind::Float) {
            file.tasks.push_back(std::move(task));
            continue;
        }
        if (version != kHqqVersion) {
            return fail(error, ErrorCode::TensorMismatch, "Invalid quantized source", {}, task.tensor);
        }
        if (!collect_hqq(segment, task, pairs[task.tensor], file, error)) {
            return false;
        }
    }
    if (cursor != file.size) {
        return fail(error, ErrorCode::IncompatibleTemplate, file.name + ": incomplete segments");
    }
    for (const auto& item : pairs) {
        const auto& pair = item.second;
        if (pair.weight == nullptr || pair.alpha == nullptr ||
            pair.weight->at("quantization") != pair.alpha->at("quantization") ||
            pair.weight->at("positive_zero") != pair.alpha->at("positive_zero")) {
            return fail(error, ErrorCode::IncompatibleTemplate, item.first + ": incomplete or inconsistent HQQ pair");
        }
        // Alpha 위치는 검증 중 확정한다. 실행 시 재검색하지 않는다.
        file.tasks[pair.task_index].alpha_offset = unsigned_value(pair.alpha->at("offset"));
    }
    return true;
}

} // namespace

bool safe_filename(const std::string& name) {
    return !name.empty() && name != "." && name != ".." && name != kOwnershipFile &&
           name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") ==
               std::string::npos;
}

bool build_conversion_plan(const std::vector<uint8_t>& bytes, uint64_t template_size, const SourceIndex& source,
                           ConversionPlan& plan, ConvertError& error) {
    try {
        const Json manifest = parse_json(bytes);
        if (manifest.at("format") != kFormatName ||
            (manifest.at("version") != kFloatVersion && manifest.at("version") != kHqqVersion) ||
            manifest.at("mnn_version") != kMnnVersion || manifest.at("template") != kTemplateFilename ||
            unsigned_value(manifest.at("template_size")) != template_size) {
            return fail(error, ErrorCode::IncompatibleTemplate, "Unsupported manifest or incorrect template size");
        }
        const auto& files = manifest.at("files");
        if (!files.is_array() || files.empty() || files.size() > kMaxOutputFiles) {
            return fail(error, ErrorCode::IncompatibleTemplate, "Invalid output file list");
        }
        ConversionPlan parsed;
        parsed.manifest_version = manifest.at("version");
        parsed.mnn_version = manifest.at("mnn_version");
        parsed.unet_quantization = manifest.value("unet_quantization", "unspecified");
        std::unordered_set<std::string> names;
        for (const auto& description : files) {
            FilePlan file;
            if (!build_file(description, template_size, parsed.manifest_version, source, file, error)) {
                return false;
            }
            if (!names.insert(file.name).second || file.size > UINT64_MAX - parsed.output_bytes) {
                return fail(error, ErrorCode::IncompatibleTemplate, "Duplicate output name or output size overflow");
            }
            parsed.output_bytes += file.size;
            for (const auto& task : file.tasks) {
                parsed.has_hqq = parsed.has_hqq || task.kind == TaskKind::Hqq;
            }
            parsed.files.push_back(std::move(file));
        }
        plan = std::move(parsed);
        error = {};
        return true;
    }
    catch (const std::bad_alloc&) {
        throw;
    }
    catch (const std::exception& exception) {
        return fail(error, ErrorCode::IncompatibleTemplate, exception.what());
    }
}

} // namespace sd15::detail
