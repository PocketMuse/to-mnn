#include "sd15/converter.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace sd15 {
namespace {

using Json = nlohmann::json;
namespace fs = std::filesystem;
constexpr uint64_t kHeaderPrefixBytes = 8;
constexpr uint64_t kMaxHeaderBytes = 4 * 1024 * 1024;
constexpr uint64_t kMaxManifestBytes = 8 * 1024 * 1024;
constexpr std::size_t kMaxChunkBytes = 4 * 1024 * 1024;
constexpr int kMaxJsonDepth = 64;
constexpr std::size_t kMaxTensorRank = 16;
constexpr std::size_t kMaxOutputFiles = 16;
constexpr std::size_t kMaxSegments = 65536;
constexpr char kFormatName[] = "sd15-mnn-template";
constexpr int kFormatVersion = 1;
constexpr char kMnnVersion[] = "3.6.1";
constexpr char kTemplateFilename[] = "graph.bin";
constexpr char kManifestFilename[] = "manifest.json";
constexpr uint32_t kBitsPerByte = 8;
constexpr uint32_t kHalfMaxFloatBits = 0x477fe000;
constexpr uint32_t kHalfSign = 0x8000;
constexpr uint32_t kHalfSignShift = 16;
constexpr uint32_t kHalfMantissa = 0x3ff;
constexpr uint32_t kHalfImplicitBit = 0x400;
constexpr uint32_t kHalfExponentMask = 0x1f;
constexpr uint32_t kFloatExponentMask = 0xff;
constexpr uint32_t kFloatSign = 0x80000000;
constexpr uint32_t kFloatMagnitude = 0x7fffffff;
constexpr uint32_t kFloatMantissa = 0x007fffff;
constexpr uint32_t kFloatImplicitBit = 0x00800000;
constexpr uint32_t kFloatExponentBits = 23;
constexpr uint32_t kHalfExponentBits = 10;
constexpr uint32_t kExponentBiasDifference = 112;
constexpr uint32_t kFirstHalfSubnormalExponent = 103;
constexpr uint32_t kMantissaShift = 13;
constexpr uint32_t kSubnormalShiftBias = 126;
static_assert(sizeof(off_t) >= 8, "64-bit file offsets are required");

bool fail(std::string& error, const std::string& message) {
    error = message;
    return false;
}

uint64_t read_le(const uint8_t* bytes, std::size_t size) {
    uint64_t value = 0;
    for (std::size_t i = 0; i < size; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (i * kBitsPerByte);
    }
    return value;
}

void write_le(uint8_t* bytes, uint32_t value, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<uint8_t>(value >> (i * kBitsPerByte));
    }
}

/// 파일 핸들을 소유하며 범위 읽기·순차 쓰기 오류를 호출자에게 전달한다.
class File {
public:
    File() = default;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File() {
        if (handle_ != nullptr) {
            std::fclose(handle_);
        }
    }

    bool open(const fs::path& path, const char* mode, std::string& error) {
        handle_ = std::fopen(path.c_str(), mode);
        return handle_ != nullptr || fail(error, "Cannot open " + path.string() + ": " + std::strerror(errno));
    }

    bool read_at(uint64_t offset, uint8_t* data, std::size_t size, std::string& error) {
        if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
            fseeko(handle_, static_cast<off_t>(offset), SEEK_SET) != 0 ||
            std::fread(data, 1, size, handle_) != size) {
            return fail(error, "Failed to read at byte " + std::to_string(offset));
        }
        return true;
    }

    bool write(const uint8_t* data, std::size_t size, std::string& error) {
        return std::fwrite(data, 1, size, handle_) == size || fail(error, "Failed to write output (check free disk space)");
    }

    bool finish(std::string& error) {
        const int result = std::fclose(handle_);
        handle_ = nullptr;
        return result == 0 || fail(error, "Failed to close output");
    }

private:
    std::FILE* handle_ = nullptr;
};

/// 임시 출력 파일을 소유하고 성공 시 디렉터리 이름을 바꿔 공개한다.
/// 오류 시 이번 작업이 만든 파일만 제거하며 재귀 삭제는 하지 않는다.
class OutputDirectory {
public:
    explicit OutputDirectory(fs::path path) : path_(std::move(path)) {}
    OutputDirectory(const OutputDirectory&) = delete;
    OutputDirectory& operator=(const OutputDirectory&) = delete;
    ~OutputDirectory() {
        if (!created_) {
            return;
        }
        std::error_code ignored;
        for (const auto& name : names_) {
            fs::remove(path_ / name, ignored);
        }
        fs::remove(path_, ignored);
    }

    bool create(std::string& error) {
        created_ = fs::create_directory(path_);
        return created_ || fail(error, "Staging directory already exists: " + path_.string());
    }

    fs::path add(const std::string& name) {
        names_.push_back(name);
        return path_ / name;
    }

    void publish(const fs::path& output) {
        if (fs::exists(output)) {
            throw std::runtime_error("Output directory already exists");
        }
        fs::rename(path_, output);
        created_ = false;
    }

private:
    fs::path path_;
    bool created_ = false;
    std::vector<std::string> names_;
};

/// 중복 key와 과도한 중첩을 거부하며 JSON을 읽는다.
Json parse_json(const std::vector<uint8_t>& bytes) {
    std::vector<std::unordered_set<std::string>> keys;
    return Json::parse(bytes.begin(), bytes.end(), [&keys](int depth, Json::parse_event_t event, Json& value) {
        if (depth > kMaxJsonDepth) {
            throw std::runtime_error("JSON nesting limit exceeded");
        }
        if (event == Json::parse_event_t::object_start) {
            keys.emplace_back();
        }
        else if (event == Json::parse_event_t::key) {
            if (!keys.back().insert(value.get<std::string>()).second) {
                throw std::runtime_error("Duplicate JSON key: " + value.get<std::string>());
            }
        } else if (event == Json::parse_event_t::object_end) {
            keys.pop_back();
        }
        return true;
    });
}

uint64_t unsigned_value(const Json& value) {
    if (!value.is_number_unsigned()) {
        throw std::runtime_error("Expected unsigned integer");
    }
    return value.get<uint64_t>();
}

uint64_t element_count(const Json& shape) {
    if (!shape.is_array() || shape.size() > kMaxTensorRank) {
        throw std::runtime_error("Invalid tensor shape");
    }
    uint64_t count = 1;
    for (const auto& dim : shape) {
        const uint64_t size = unsigned_value(dim);
        if (size != 0 && count > std::numeric_limits<uint64_t>::max() / size) {
            throw std::runtime_error("Tensor shape overflow");
        }
        count *= size;
    }
    return count;
}

std::size_t dtype_width(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") {
        return 4;
    }
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") {
        return 2;
    }
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") {
        return 8;
    }
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") {
        return 1;
    }
    throw std::runtime_error("Unsupported dtype: " + dtype);
}

/// 원본 텐서의 파일 시작 기준 offset과 원소 수·dtype·shape.
struct SourceTensor {
    uint64_t offset;
    uint64_t elements;
    std::string dtype;
    Json shape;
};
using SourceIndex = std::unordered_map<std::string, SourceTensor>;

/// 가중치 본문을 읽지 않고 safetensors 헤더와 데이터 범위를 검증한다.
bool read_source(File& file, uint64_t file_size, SourceIndex& index, std::string& error) {
    std::array<uint8_t, kHeaderPrefixBytes> prefix{};
    if (!file.read_at(0, prefix.data(), prefix.size(), error)) {
        return false;
    }
    const uint64_t header_size = read_le(prefix.data(), prefix.size());
    if (header_size == 0 || header_size > kMaxHeaderBytes ||
        file_size < kHeaderPrefixBytes || header_size > file_size - kHeaderPrefixBytes) {
        return fail(error, "Invalid or oversized safetensors header");
    }
    std::vector<uint8_t> bytes(static_cast<std::size_t>(header_size));
    if (!file.read_at(kHeaderPrefixBytes, bytes.data(), bytes.size(), error)) {
        return false;
    }
    if (bytes.front() != '{') {
        return fail(error, "Safetensors header must start with an object");
    }
    const Json header = parse_json(bytes);
    const uint64_t data_start = kHeaderPrefixBytes + header_size;
    const uint64_t data_size = file_size - data_start;
    std::vector<std::pair<uint64_t, uint64_t>> intervals;
    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") {
            if (!it->is_object()) {
                return fail(error, "Invalid safetensors metadata");
            }
            for (const auto& value : *it) {
                if (!value.is_string()) {
                    return fail(error, "Safetensors metadata values must be strings");
                }
            }
            continue;
        }
        const auto& tensor = it.value();
        const auto& offsets = tensor.at("data_offsets");
        if (!offsets.is_array() || offsets.size() != 2) {
            return fail(error, it.key() + ": invalid data_offsets");
        }
        const uint64_t begin = unsigned_value(offsets[0]);
        const uint64_t end = unsigned_value(offsets[1]);
        const std::string dtype = tensor.at("dtype");
        const uint64_t count = element_count(tensor.at("shape"));
        const std::size_t width = dtype_width(dtype);
        if (begin > end || end > data_size || count > std::numeric_limits<uint64_t>::max() / width ||
            end - begin != count * width) {
            return fail(error, it.key() + ": invalid shape or data range");
        }
        intervals.emplace_back(begin, end);
        index.emplace(it.key(), SourceTensor{data_start + begin, count, dtype, tensor.at("shape")});
    }
    std::sort(intervals.begin(), intervals.end());
    uint64_t cursor = 0;
    for (const auto& interval : intervals) {
        if (interval.first != cursor) {
            return fail(error, "Safetensors data contains overlaps or holes");
        }
        cursor = interval.second;
    }
    return cursor == data_size || fail(error, "Unindexed safetensors data");
}

bool safe_filename(const std::string& name) {
    return !name.empty() && name != "." && name != ".." &&
           name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") == std::string::npos;
}

/// 출력 구간 배치와 원본 key·shape·dtype, 템플릿 참조 범위를 검증한다.
bool validate_manifest(const Json& manifest, uint64_t template_size, const SourceIndex& source, std::string& error) {
    if (manifest.at("format") != kFormatName || manifest.at("version") != kFormatVersion ||
        manifest.at("mnn_version") != kMnnVersion || manifest.at("template") != kTemplateFilename ||
        unsigned_value(manifest.at("template_size")) != template_size) {
        return fail(error, "Unsupported manifest or incorrect template size");
    }
    const auto& files = manifest.at("files");
    if (!files.is_array() || files.empty() || files.size() > kMaxOutputFiles) {
        return fail(error, "Invalid output file list");
    }
    std::unordered_set<std::string> names;
    for (const auto& file : files) {
        const std::string name = file.at("name");
        const uint64_t file_size = unsigned_value(file.at("size"));
        if (!safe_filename(name) || !names.insert(name).second ||
            file_size > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
            return fail(error, "Invalid output filename or size: " + name);
        }
        const auto& segments = file.at("segments");
        if (!segments.is_array() || segments.empty() || segments.size() > kMaxSegments) {
            return fail(error, name + ": invalid segments");
        }
        uint64_t cursor = 0;
        for (const auto& segment : segments) {
            const uint64_t offset = unsigned_value(segment.at("offset"));
            const uint64_t size = unsigned_value(segment.at("size"));
            const std::string kind = segment.at("kind");
            if (offset != cursor || size == 0 || size > file_size - cursor) {
                return fail(error, name + ": segment overlap, gap, or size mismatch");
            }
            cursor += size;
            if (kind == "zero") {
                continue;
            }
            if (kind == "literal") {
                const uint64_t start = unsigned_value(segment.at("template_offset"));
                if (start > template_size || size > template_size - start) {
                    return fail(error, name + ": literal outside template");
                }
                continue;
            }
            if (kind != "tensor") {
                return fail(error, name + ": unknown segment kind");
            }
            const std::string key = segment.at("key");
            const auto found = source.find(key);
            if (found == source.end()) {
                return fail(error, "Missing tensor: " + key);
            }
            const auto& tensor = found->second;
            const std::string dtype = segment.at("dtype");
            if ((dtype != "F16" && dtype != "F32") || (tensor.dtype != "F16" && tensor.dtype != "F32")) {
                return fail(error, key + ": only F16/F32 weights are supported");
            }
            const std::size_t width = dtype_width(dtype);
            if (tensor.shape != segment.at("source_shape") ||
                element_count(segment.at("shape")) != tensor.elements ||
                size % width != 0 || size / width != tensor.elements ||
                !segment.at("positive_zero").is_boolean()) {
                return fail(error, key + ": tensor shape/dtype/byte count mismatch");
            }
        }
        if (cursor != file_size) {
            return fail(error, name + ": incomplete segments");
        }
    }
    return true;
}

/// subnormal과 signed zero를 포함해 FP16을 FP32 비트로 확장한다.
uint32_t half_to_float_bits(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & kHalfSign) << kHalfSignShift;
    uint32_t exponent = (half >> kHalfExponentBits) & kHalfExponentMask;
    uint32_t mantissa = half & kHalfMantissa;
    if (exponent == 0 && mantissa == 0) {
        return sign;
    }
    if (exponent == 0) {
        int shift = 0;
        while ((mantissa & kHalfImplicitBit) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        exponent = kExponentBiasDifference + 1 - shift;
        return sign | (exponent << kFloatExponentBits) | ((mantissa & kHalfMantissa) << kMantissaShift);
    }
    exponent = exponent == kHalfExponentMask ? kFloatExponentMask : exponent + kExponentBiasDifference;
    return sign | (exponent << kFloatExponentBits) | (mantissa << kMantissaShift);
}

/// 유한 FP32를 MNN 규칙에 따라 범위 제한 후 0 방향으로 FP16 절삭한다.
uint16_t float_to_half_bits(uint32_t bits) {
    const uint16_t sign = static_cast<uint16_t>((bits & kFloatSign) >> kHalfSignShift);
    const uint32_t magnitude = std::min(bits & kFloatMagnitude, kHalfMaxFloatBits);
    const uint32_t exponent = magnitude >> kFloatExponentBits;
    if (exponent < kFirstHalfSubnormalExponent) {
        return sign;
    }
    if (exponent <= kExponentBiasDifference) {
        return sign | static_cast<uint16_t>(((magnitude & kFloatMantissa) | kFloatImplicitBit) >> (kSubnormalShiftBias - exponent));
    }
    return sign | static_cast<uint16_t>(((exponent - kExponentBiasDifference) << kHalfExponentBits) |
                                       ((magnitude & kFloatMantissa) >> kMantissaShift));
}

/// 전달받은 버퍼를 재사용해 텐서를 변환·기록하며 비유한 값은 거부한다.
bool write_tensor(
    File& source,
    File& output,
    const SourceTensor& tensor,
    const Json& segment,
    std::vector<uint8_t>& input,
    std::vector<uint8_t>& converted,
    std::string& error
) {
    const std::size_t input_width = dtype_width(tensor.dtype);
    const std::size_t output_width = dtype_width(segment.at("dtype"));
    const bool positive_zero = segment.at("positive_zero");
    uint64_t cursor = 0;
    while (cursor < tensor.elements) {
        const std::size_t count = static_cast<std::size_t>(std::min<uint64_t>(tensor.elements - cursor, input.size() / input_width));
        if (!source.read_at(tensor.offset + cursor * input_width, input.data(), count * input_width, error)) {
            return false;
        }
        for (std::size_t i = 0; i < count; ++i) {
            uint32_t bits = static_cast<uint32_t>(read_le(input.data() + i * input_width, input_width));
            if (input_width == 2) {
                bits = half_to_float_bits(static_cast<uint16_t>(bits));
            }
            if (((bits >> kFloatExponentBits) & kFloatExponentMask) == kFloatExponentMask) {
                return fail(error, "Non-finite weight at element " + std::to_string(cursor + i));
            }
            if (positive_zero && (bits & kFloatMagnitude) == 0) {
                bits = 0;
            }
            const uint32_t value = output_width == 2 ? float_to_half_bits(bits) : bits;
            write_le(converted.data() + i * output_width, value, output_width);
        }
        if (!output.write(converted.data(), count * output_width, error)) {
            return false;
        }
        cursor += count;
    }
    return true;
}

/// 입력을 검증하고 literal·tensor·zero 구간을 조립한 뒤 완성 파일을 공개한다.
bool run(const ConvertOptions& options, std::string& error) {
    if (options.chunk_bytes < 4 || options.chunk_bytes > kMaxChunkBytes || options.chunk_bytes % 4 != 0) {
        return fail(error, "chunk-bytes must be a multiple of 4 in [4, 4194304]");
    }
    if (fs::exists(options.output_dir)) {
        return fail(error, "Output directory already exists: " + options.output_dir);
    }
    File checkpoint;
    File template_file;
    File manifest_file;
    const fs::path template_dir(options.template_dir);
    if (!checkpoint.open(options.checkpoint, "rb", error) ||
        !template_file.open(template_dir / kTemplateFilename, "rb", error) ||
        !manifest_file.open(template_dir / kManifestFilename, "rb", error)) {
        return false;
    }
    SourceIndex source;
    if (!read_source(checkpoint, fs::file_size(options.checkpoint), source, error)) {
        return false;
    }
    const uint64_t manifest_size = fs::file_size(template_dir / kManifestFilename);
    if (manifest_size == 0 || manifest_size > kMaxManifestBytes) {
        return fail(error, "Invalid or oversized manifest");
    }
    std::vector<uint8_t> manifest_bytes(static_cast<std::size_t>(manifest_size));
    if (!manifest_file.read_at(0, manifest_bytes.data(), manifest_bytes.size(), error)) {
        return false;
    }
    const Json manifest = parse_json(manifest_bytes);
    if (!validate_manifest(manifest, fs::file_size(template_dir / kTemplateFilename), source, error)) {
        return false;
    }
    std::vector<uint8_t> input(options.chunk_bytes);
    std::vector<uint8_t> converted(options.chunk_bytes * 2);
    OutputDirectory staging(options.output_dir + ".partial");
    if (!staging.create(error)) {
        return false;
    }
    for (const auto& file : manifest.at("files")) {
        const std::string name = file.at("name");
        File output;
        if (!output.open(staging.add(name), "wbx", error)) {
            return false;
        }
        for (const auto& segment : file.at("segments")) {
            const std::string kind = segment.at("kind");
            if (kind == "tensor") {
                const std::string key = segment.at("key");
                if (!write_tensor(checkpoint, output, source.at(key), segment, input, converted, error)) {
                    error = key + ": " + error;
                    return false;
                }
                continue;
            }
            if (kind == "zero") {
                std::fill(input.begin(), input.end(), 0);
            }
            const uint64_t size = unsigned_value(segment.at("size"));
            const uint64_t start = kind == "literal" ? unsigned_value(segment.at("template_offset")) : 0;
            uint64_t cursor = 0;
            while (cursor < size) {
                const std::size_t count = static_cast<std::size_t>(std::min<uint64_t>(size - cursor, input.size()));
                if (kind == "literal" && !template_file.read_at(start + cursor, input.data(), count, error)) {
                    return false;
                }
                if (!output.write(input.data(), count, error)) {
                    return false;
                }
                cursor += count;
            }
        }
        if (!output.finish(error)) {
            return false;
        }
    }
    staging.publish(options.output_dir);
    return true;
}

}  // namespace

bool convert(const ConvertOptions& options, std::string& error) {
    error.clear();
    // JSON/파일시스템/할당 예외가 라이브러리 경계를 넘지 않게 한다.
    try {
        return run(options, error);
    }
    catch (const std::exception& exception) {
        return fail(error, exception.what());
    }
}

}  // namespace sd15
