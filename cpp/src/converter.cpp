#include "sd15/converter.hpp"
#include "sd15/hqq.hpp"
#include "conversion_progress.hpp"
#include "tensor_transform.hpp"
#include "conversion_error.hpp"
#include "conversion_json.hpp"
#include "conversion_plan.hpp"

#include <algorithm>
#include <array>
#include <mutex>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unordered_set>
#include <vector>

namespace sd15 {
namespace {

namespace fs = std::filesystem;
constexpr uint64_t kHeaderPrefixBytes = 8;
constexpr uint64_t kMaxHeaderBytes = 4 * 1024 * 1024;
constexpr uint64_t kMaxManifestBytes = 8 * 1024 * 1024;
constexpr std::size_t kMaxChunkBytes = 4 * 1024 * 1024;
constexpr char kManifestFilename[] = "manifest.json";
constexpr uint32_t kBitsPerByte = 8;
static_assert(sizeof(off_t) >= 8, "64-bit file offsets are required");
std::mutex conversion_mutex;
using namespace detail;

uint64_t read_le(const uint8_t* bytes, std::size_t size) {
    uint64_t value = 0;
    for (std::size_t i = 0; i < size; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (i * kBitsPerByte);
    }
    return value;
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

    bool open(const fs::path& path, const char* mode, ConvertError& error) {
        path_ = path.string();
        handle_ = std::fopen(path.c_str(), mode);
        if (handle_ == nullptr) {
            return fail(error, mode[0] == 'r' ? ErrorCode::ReadFailed : ErrorCode::WriteFailed,
                        "Cannot open " + path_ + ": " + std::strerror(errno), path_);
        }
        return true;
    }

    bool read_at(uint64_t offset, uint8_t* data, std::size_t size, ConvertError& error) {
        if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
            fseeko(handle_, static_cast<off_t>(offset), SEEK_SET) != 0 ||
            std::fread(data, 1, size, handle_) != size) {
            return fail(error, ErrorCode::ReadFailed, "Failed to read at byte " + std::to_string(offset), path_);
        }
        return true;
    }

    bool seek(uint64_t offset, ConvertError& error) {
        if (offset <= static_cast<uint64_t>(std::numeric_limits<off_t>::max()) &&
            fseeko(handle_, static_cast<off_t>(offset), SEEK_SET) == 0) {
            return true;
        }
        return fail(error, ErrorCode::WriteFailed, "Failed to seek output", path_);
    }

    bool write_at(uint64_t offset, const uint8_t* data, std::size_t size, ConvertError& error) {
        return seek(offset, error) && write(data, size, error);
    }

    bool write(const uint8_t* data, std::size_t size, ConvertError& error) {
        if (std::fwrite(data, 1, size, handle_) == size) {
            return true;
        }
        return fail(error, errno == ENOSPC ? ErrorCode::InsufficientSpace : ErrorCode::WriteFailed,
                    "Failed to write output (check free disk space)", path_);
    }

    bool finish(ConvertError& error) {
        const int result = std::fclose(handle_);
        handle_ = nullptr;
        return result == 0 || fail(error, ErrorCode::WriteFailed, "Failed to close output", path_);
    }

private:
    std::FILE* handle_ = nullptr;
    std::string path_;
};

/// 임시 출력 파일을 소유하고 성공 시 디렉터리 이름을 바꿔 공개한다.
/// 오류 시 이번 작업이 만든 파일만 제거하며 재귀 삭제는 하지 않는다.
class OutputDirectory {
public:
    explicit OutputDirectory(fs::path path) : path_(std::move(path)) {}
    OutputDirectory(const OutputDirectory&) = delete;
    OutputDirectory& operator=(const OutputDirectory&) = delete;
    ~OutputDirectory() {
        cleanup();
    }

    bool cleanup() noexcept {
        if (!created_) {
            return true;
        }
        std::error_code error;
        bool ok = true;
        for (const auto& name : names_) {
            if (name == kOwnershipFile) {
                continue;
            }
            try {
                fs::remove(path_ / name, error);
                ok = ok && !error;
            }
            catch (...) {
                ok = false;
            }
        }
        if (ok) {
            try {
                fs::remove(path_ / kOwnershipFile, error);
                ok = !error;
            }
            catch (...) {
                ok = false;
            }
        }
        if (ok) {
            fs::remove(path_, error);
            ok = !error;
        }
        created_ = !ok;
        return ok;
    }

    bool create(ConvertError& error) {
        std::error_code ec;
        created_ = fs::create_directory(path_, ec);
        return created_ || fail(error, ErrorCode::WriteFailed,
                               ec ? ec.message() : "Staging directory already exists", path_.string());
    }

    fs::path add(const std::string& name) {
        names_.push_back(name);
        return path_ / name;
    }

    bool publish(const fs::path& output, ConvertError& error) {
        std::error_code ec;
        const bool exists = fs::exists(output, ec);
        if (ec || exists) {
            return fail(error, ErrorCode::WriteFailed,
                        ec ? ec.message() : "Output directory already exists", output.string());
        }
        fs::rename(path_, output, ec);
        if (ec) {
            return fail(error, ErrorCode::WriteFailed, ec.message(), output.string());
        }
        created_ = false;
        try {
            std::error_code ignored;
            fs::remove(output / kOwnershipFile, ignored);
        }
        catch (...) {
            // 공개는 완료됐다. 표식 제거 실패는 결과를 되돌리지 않는다.
        }
        return true;
    }

private:
    fs::path path_;
    bool created_ = false;
    std::vector<std::string> names_;
};

/// 가중치 본문을 읽지 않고 safetensors 헤더와 데이터 범위를 검증한다.
bool read_source(File& file, uint64_t file_size, SourceIndex& index, ConvertError& error) {
    try {
        std::array<uint8_t, kHeaderPrefixBytes> prefix{};
        if (!file.read_at(0, prefix.data(), prefix.size(), error)) {
            return false;
        }
        const uint64_t header_size = read_le(prefix.data(), prefix.size());
        if (header_size == 0 || header_size > kMaxHeaderBytes ||
            file_size < kHeaderPrefixBytes || header_size > file_size - kHeaderPrefixBytes) {
            return fail(error, ErrorCode::InvalidInput, "Invalid or oversized safetensors header");
        }
        std::vector<uint8_t> bytes(static_cast<std::size_t>(header_size));
        if (!file.read_at(kHeaderPrefixBytes, bytes.data(), bytes.size(), error)) {
            return false;
        }
        if (bytes.front() != '{') {
            return fail(error, ErrorCode::InvalidInput, "Safetensors header must start with an object");
        }
        const Json header = parse_json(bytes);
        const uint64_t data_start = kHeaderPrefixBytes + header_size;
        const uint64_t data_size = file_size - data_start;
        std::vector<std::pair<uint64_t, uint64_t>> intervals;
        for (auto it = header.begin(); it != header.end(); ++it) {
            if (it.key() == "__metadata__") {
                if (!it->is_object()) {
                    return fail(error, ErrorCode::InvalidInput, "Invalid safetensors metadata");
                }
                for (const auto& value : *it) {
                    if (!value.is_string()) {
                        return fail(error, ErrorCode::InvalidInput, "Safetensors metadata values must be strings");
                    }
                }
                continue;
            }
            const auto& tensor = it.value();
            const auto& offsets = tensor.at("data_offsets");
            if (!offsets.is_array() || offsets.size() != 2) {
                return fail(error, ErrorCode::InvalidInput, it.key() + ": invalid data_offsets");
            }
            const uint64_t begin = unsigned_value(offsets[0]);
            const uint64_t end = unsigned_value(offsets[1]);
            const std::string dtype = tensor.at("dtype");
            const uint64_t count = element_count(tensor.at("shape"));
            const std::size_t width = dtype_width(dtype);
            if (begin > end || end > data_size || count > std::numeric_limits<uint64_t>::max() / width ||
                end - begin != count * width) {
                return fail(error, ErrorCode::InvalidInput, it.key() + ": invalid shape or data range");
            }
            intervals.emplace_back(begin, end);
            index.emplace(it.key(), SourceTensor{data_start + begin, count, dtype, tensor.at("shape").get<std::vector<uint64_t>>()});
        }
        std::sort(intervals.begin(), intervals.end());
        uint64_t cursor = 0;
        for (const auto& interval : intervals) {
            if (interval.first != cursor) {
                return fail(error, ErrorCode::InvalidInput, "Safetensors data contains overlaps or holes");
            }
            cursor = interval.second;
        }
        return cursor == data_size || fail(error, ErrorCode::InvalidInput, "Unindexed safetensors data");
    }
    catch (const std::bad_alloc&) {
        throw;
    }
    catch (const std::exception& exception) {
        return fail(error, ErrorCode::InvalidInput, exception.what());
    }
}

/// 전달받은 버퍼를 재사용해 텐서를 변환·기록하며 비유한 값은 거부한다.
bool write_tensor(
    File& source,
    File& output,
    const ConversionTask& task,
    std::vector<uint8_t>& input,
    std::vector<uint8_t>& converted,
    Progress& progress,
    ConvertError& error
) {
    const std::size_t input_width = storage_width(task.source_storage);
    const std::size_t output_width = storage_width(task.target_storage);
    uint64_t cursor = 0;
    while (cursor < task.elements) {
        progress.check();
        const auto begin = Clock::now();
        const std::size_t count = static_cast<std::size_t>(std::min<uint64_t>(task.elements - cursor, input.size() / input_width));
        if (!source.read_at(task.source_offset + cursor * input_width, input.data(), count * input_width, error)) {
            return false;
        }
        const auto invalid = encode_float_chunk(input.data(), count, task.source_storage,
                                                task.target_storage, task.positive_zero, converted.data());
        if (invalid) {
            return fail(error, ErrorCode::NonFiniteWeight, "Non-finite weight at element " + std::to_string(cursor + *invalid));
        }
        if (!output.write(converted.data(), count * output_width, error)) {
            return false;
        }
        cursor += count;
        progress.advance(WorkKind::Float, count * output_width, begin);
    }
    return true;
}

/// 그룹 배치만 메모리에 올리고 payload와 alpha를 각각의 출력 위치에 기록한다.
bool write_quantized(
    File& source, File& output, const ConversionTask& task,
    std::vector<uint8_t>& input, int iterations,
    Progress& progress, ConvertError& error
) {
    const std::size_t area = task.group_elements;
    const uint64_t groups = task.group_count;
    const std::size_t width = storage_width(task.source_storage);
    const std::size_t batch_groups = std::max<std::size_t>(1, input.size() / (area * width));
    std::vector<float> values(batch_groups * area);
    std::vector<uint8_t> payload(batch_groups * area);
    std::vector<uint8_t> scales(batch_groups * kHqqAlphaBytes);
    for (uint64_t group = 0; group < groups; group += batch_groups) {
        progress.check();
        const auto begin = Clock::now();
        const std::size_t count = static_cast<std::size_t>(std::min<uint64_t>(groups - group, batch_groups));
        const std::size_t elements = count * area;
        for (std::size_t cursor = 0; cursor < elements;) {
            progress.check();
            const std::size_t read_count = std::min(elements - cursor, input.size() / width);
            if (!source.read_at(task.source_offset + (group * area + cursor) * width, input.data(), read_count * width, error)) {
                return false;
            }
            const auto invalid = decode_float_chunk(input.data(), read_count, task.source_storage,
                                                    task.positive_zero, values.data() + cursor);
            if (invalid) {
                return fail(error, ErrorCode::NonFiniteWeight, "Non-finite HQQ weight");
            }
            cursor += read_count;
        }
        const auto compute_begin = Clock::now();
        for (std::size_t i = 0; i < count; ++i) {
            progress.check();
            progress.emit();
            const auto hqq_error = encode_hqq_group(values.data() + i * area, area, iterations,
                                                   payload.data() + i * area, scales.data() + i * kHqqAlphaBytes);
            if (hqq_error != HqqError::None) {
                return fail(error, ErrorCode::InvalidInput, hqq_error_message(hqq_error));
            }
        }
        progress.hqq_compute_time += Clock::now() - compute_begin;
        if (!output.write_at(task.output_offset + group * area, payload.data(), elements, error) ||
            !output.write_at(task.alpha_offset + group * kHqqAlphaBytes, scales.data(), count * kHqqAlphaBytes, error)) {
            return false;
        }
        progress.advance(WorkKind::Hqq, elements + count * kHqqAlphaBytes, begin);
    }
    return true;
}

/// 쓰기 없이 입력과 템플릿을 검증하고 출력 계획을 만든다.
bool prepare(const ConvertOptions& options, File& checkpoint, File& template_file,
             ConversionPlan& plan, InspectResult& info, ConvertError& error) {
    try {
        if (options.hqq_iterations < 0 || options.hqq_iterations > kHqqIterations) {
            return fail(error, ErrorCode::InvalidInput, "HQQ iterations must be in [0, 20]");
        }
        if (options.checkpoint.empty() || options.template_dir.empty() || options.output_dir.empty()) {
            return fail(error, ErrorCode::InvalidInput, "Checkpoint, template and output paths are required");
        }
        if (options.chunk_bytes < 4 || options.chunk_bytes > kMaxChunkBytes || options.chunk_bytes % 4 != 0) {
            return fail(error, ErrorCode::InvalidInput, "chunk-bytes must be a multiple of 4 in [4, 4194304]");
        }
        if (fs::exists(options.output_dir)) {
            return fail(error, ErrorCode::OutputExists, "Output directory already exists", options.output_dir);
        }
        const fs::path parent = fs::absolute(options.output_dir).parent_path();
        if (!fs::is_directory(parent)) {
            return fail(error, ErrorCode::InvalidInput, "Output parent directory must exist", parent.string());
        }
        info.partial_exists = fs::symlink_status(options.output_dir + ".partial").type() != fs::file_type::not_found;
        File manifest_file;
        const fs::path template_dir(options.template_dir);
        if (!checkpoint.open(options.checkpoint, "rb", error) ||
            !template_file.open(template_dir / kTemplateFilename, "rb", error) ||
            !manifest_file.open(template_dir / kManifestFilename, "rb", error)) {
            return false;
        }
        std::error_code ec;
        const auto source_size = fs::file_size(options.checkpoint, ec);
        if (ec) {
            return fail(error, ErrorCode::ReadFailed, ec.message(), options.checkpoint);
        }
        SourceIndex source;
        if (!read_source(checkpoint, source_size, source, error)) {
            error.path = options.checkpoint;
            return false;
        }
        const auto manifest_path = template_dir / kManifestFilename;
        const auto manifest_size = fs::file_size(manifest_path, ec);
        if (ec) {
            return fail(error, ErrorCode::ReadFailed, ec.message(), manifest_path.string());
        }
        if (manifest_size == 0 || manifest_size > kMaxManifestBytes) {
            return fail(error, ErrorCode::IncompatibleTemplate, "Invalid or oversized manifest", manifest_path.string());
        }
        const auto template_size = fs::file_size(template_dir / kTemplateFilename, ec);
        if (ec) {
            return fail(error, ErrorCode::ReadFailed, ec.message(), (template_dir / kTemplateFilename).string());
        }
        {
            std::vector<uint8_t> manifest_bytes(static_cast<std::size_t>(manifest_size));
            if (!manifest_file.read_at(0, manifest_bytes.data(), manifest_bytes.size(), error)) {
                return false;
            }
            if (!build_conversion_plan(manifest_bytes, template_size, source, plan, error)) {
                error.path = manifest_path.string();
                return false;
            }
        }
        info.manifest_version = plan.manifest_version;
        info.mnn_version = plan.mnn_version;
        info.unet_quantization = plan.unet_quantization;
        info.output_bytes = plan.output_bytes;
        if (plan.has_hqq) {
            info.hqq_iterations = options.hqq_iterations;
        }
        for (const auto& file : plan.files) {
            info.files.push_back({file.name, file.size});
        }
        const auto space = fs::space(parent, ec);
        if (!ec && space.available != static_cast<uintmax_t>(-1)) {
            info.available_bytes = space.available;
            if (space.available < info.output_bytes) {
                return fail(error, ErrorCode::InsufficientSpace, "Insufficient output disk space", parent.string());
            }
        }
        error = {};
        info.ok = true;
        return true;
    }
    catch (const fs::filesystem_error& exception) {
        return fail(error, ErrorCode::InvalidInput, exception.what(), exception.path1().string());
    }
}

bool write_task(File& checkpoint, File& template_file, File& output,
                const ConversionTask& task, std::vector<uint8_t>& input,
                std::vector<uint8_t>& converted, int iterations,
                Progress& progress, ConvertError& error) {
    if (task.kind == TaskKind::Hqq) {
        return write_quantized(checkpoint, output, task, input, iterations, progress, error);
    }
    if (!output.seek(task.output_offset, error)) {
        return false;
    }
    if (task.kind == TaskKind::Float) {
        return write_tensor(checkpoint, output, task, input, converted, progress, error);
    }
    if (task.kind == TaskKind::Zero) {
        std::fill(input.begin(), input.end(), 0);
    }
    uint64_t cursor = 0;
    while (cursor < task.size) {
        progress.check();
        const auto begin = Clock::now();
        const std::size_t count = static_cast<std::size_t>(std::min<uint64_t>(task.size - cursor, input.size()));
        if (task.kind == TaskKind::Literal &&
            !template_file.read_at(task.source_offset + cursor, input.data(), count, error)) {
            return false;
        }
        if (!output.write(input.data(), count, error)) {
            return false;
        }
        cursor += count;
        progress.advance(WorkKind::Copy, count, begin);
    }
    return true;
}

bool write_owner(OutputDirectory& staging, const ConversionPlan& plan, ConvertError& error) {
    Json owner = {{"format", kFormatName}, {"files", Json::array()}};
    for (const auto& file : plan.files) {
        owner["files"].push_back(file.name);
    }
    File marker;
    const std::string bytes = owner.dump();
    return marker.open(staging.add(kOwnershipFile), "wbx", error) &&
           marker.write(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), error) &&
           marker.finish(error);
}

void run(const ConvertOptions& options, Progress& progress, OutputDirectory& staging,
         ConvertResult& result) {
    auto& error = result.error;
    progress.check();
    progress.emit(true);
    File checkpoint;
    File template_file;
    ConversionPlan plan;
    InspectResult info;
    if (!prepare(options, checkpoint, template_file, plan, info, error)) {
        return;
    }
    progress.check();
    if (info.partial_exists) {
        fail(error, ErrorCode::StaleOutput, "Staging directory already exists", options.output_dir + ".partial");
        return;
    }
    progress.value.total_bytes = info.output_bytes;
    result.hqq_iterations = info.hqq_iterations;
    for (const auto& file : plan.files) {
        for (const auto& task : file.tasks) {
            const auto work = task.kind == TaskKind::Float ? WorkKind::Float :
                (task.kind == TaskKind::Hqq ? WorkKind::Hqq : WorkKind::Copy);
            const uint64_t bytes = task.size +
                (task.kind == TaskKind::Hqq ? task.group_count * kHqqAlphaBytes : 0);
            progress.total[static_cast<std::size_t>(work)] += bytes;
        }
    }
    std::vector<uint8_t> input(options.chunk_bytes);
    std::vector<uint8_t> converted(options.chunk_bytes * 2);
    if (!staging.create(error) || !write_owner(staging, plan, error)) {
        return;
    }
    progress.value.stage = ConvertStage::Converting;
    for (const auto& file : plan.files) {
        progress.value.component = file.name.substr(0, file.name.find('.'));
        progress.emit(true);
        progress.check();
        File output;
        if (!output.open(staging.add(file.name), "wbx", error)) {
            return;
        }
        for (const auto& task : file.tasks) {
            progress.check();
            if (!write_task(checkpoint, template_file, output, task, input, converted,
                            options.hqq_iterations, progress, error)) {
                error.tensor = task.tensor;
                if (!task.tensor.empty()) {
                    if (error.path.empty()) {
                        error.path = options.checkpoint;
                    }
                    error.message = task.tensor + ": " + error.message;
                }
                return;
            }
        }
        if (!output.finish(error)) {
            return;
        }
    }
    progress.value.stage = ConvertStage::Finalizing;
    progress.emit(true);
    progress.check();
    result.files = std::move(info.files);
    if (!staging.publish(options.output_dir, error)) {
        return;
    }
    result.status = ConvertStatus::Succeeded;
    error = {};
    progress.value.stage = ConvertStage::Completed;
    try {
        progress.emit(true);
    }
    catch (...) {
    }
}

}  // namespace

InspectResult inspect(const ConvertOptions& options) {
    InspectResult result;
    try {
        File checkpoint;
        File template_file;
        ConversionPlan plan;
        prepare(options, checkpoint, template_file, plan, result, result.error);
    }
    catch (const std::bad_alloc&) {
        result.error.code = ErrorCode::OutOfMemory;
        result.error.message.clear();
    }
    catch (const std::exception& exception) {
        capture_exception(result.error, ErrorCode::InternalError, exception);
    }
    catch (...) {
        result.error.code = ErrorCode::InternalError;
    }
    return result;
}

ConvertResult convert(const ConvertOptions& options, const ProgressCallback& on_progress,
                      const CancellationToken& cancellation) {
    ConvertResult result;
    Progress progress{on_progress, cancellation};
    std::optional<OutputDirectory> staging;
    std::unique_lock<std::mutex> lock(conversion_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        result.error.code = ErrorCode::Busy;
        return result;
    }
    try {
        staging.emplace(options.output_dir + ".partial");
        run(options, progress, *staging, result);
    }
    catch (const Cancelled&) {
        result.status = ConvertStatus::Cancelled;
        result.error = {};
    }
    catch (const CallbackFailed&) {
        result.error.code = ErrorCode::CallbackFailed;
        result.error.message.clear();
    }
    catch (const std::bad_alloc&) {
        result.error.code = ErrorCode::OutOfMemory;
        result.error.message.clear();
    }
    catch (const std::exception& exception) {
        capture_exception(result.error, ErrorCode::InternalError, exception);
    }
    catch (...) {
        result.error.code = ErrorCode::InternalError;
    }
    if (staging && !staging->cleanup()) {
        result.status = ConvertStatus::Failed;
        result.error.code = ErrorCode::CleanupFailed;
    }
    if (result.status != ConvertStatus::Succeeded) {
        result.files.clear();
    }
    result.elapsed_ms = progress.elapsed();
    result.hqq_compute_ms = progress.hqq_compute_time.count() * 1000.0;
    return result;
}

CleanupResult cleanup_partial(const ConvertOptions& options) {
    CleanupResult result;
    std::unique_lock<std::mutex> lock(conversion_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        result.error.code = ErrorCode::Busy;
        return result;
    }
    try {
        if (options.output_dir.empty()) {
            result.error.code = ErrorCode::InvalidInput;
            return result;
        }
        const fs::path path(options.output_dir + ".partial");
        const auto type = fs::symlink_status(path).type();
        if (type == fs::file_type::not_found) {
            result.ok = true;
            result.error = {};
            return result;
        }
        if (type != fs::file_type::directory ||
            fs::symlink_status(path / kOwnershipFile).type() != fs::file_type::regular) {
            fail(result.error, ErrorCode::CleanupFailed, "Refusing unowned staging directory", path.string());
            return result;
        }
        constexpr uint64_t kMaxOwnershipBytes = 8192;
        const auto size = fs::file_size(path / kOwnershipFile);
        if (size == 0 || size > kMaxOwnershipBytes) {
            fail(result.error, ErrorCode::CleanupFailed, "Invalid ownership marker", path.string());
            return result;
        }
        File marker;
        std::vector<uint8_t> bytes(size);
        if (!marker.open(path / kOwnershipFile, "rb", result.error) ||
            !marker.read_at(0, bytes.data(), bytes.size(), result.error) || !marker.finish(result.error)) {
            return result;
        }
        const Json owner = parse_json(bytes);
        if (owner.at("format") != kFormatName || !owner.at("files").is_array() ||
            owner.at("files").empty() || owner.at("files").size() > kMaxOutputFiles) {
            fail(result.error, ErrorCode::CleanupFailed, "Invalid ownership marker", path.string());
            return result;
        }
        std::unordered_set<std::string> allowed;
        for (const auto& item : owner.at("files")) {
            const std::string name = item;
            if (!safe_filename(name) || !allowed.insert(name).second) {
                fail(result.error, ErrorCode::CleanupFailed, "Invalid owned filename", path.string());
                return result;
            }
        }
        allowed.insert(kOwnershipFile);
        // 모두 검증한 뒤 삭제한다. 링크·하위 폴더·알 수 없는 파일은 보존한다.
        for (const auto& entry : fs::directory_iterator(path)) {
            if (!allowed.count(entry.path().filename().string()) ||
                entry.symlink_status().type() != fs::file_type::regular) {
                fail(result.error, ErrorCode::CleanupFailed, "Staging contains unexpected files", path.string());
                return result;
            }
        }
        for (const auto& name : allowed) {
            if (name != kOwnershipFile) {
                fs::remove(path / name);
            }
        }
        fs::remove(path / kOwnershipFile);
        fs::remove(path);
        result.ok = true;
        result.error = {};
    }
    catch (const std::bad_alloc&) {
        result.error.code = ErrorCode::OutOfMemory;
        result.error.message.clear();
    }
    catch (const std::exception& exception) {
        capture_exception(result.error, ErrorCode::CleanupFailed, exception);
    }
    catch (...) {
        result.error.code = ErrorCode::CleanupFailed;
    }
    return result;
}

const char* error_code_name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::None: return "NONE";
        case ErrorCode::InvalidInput: return "INVALID_INPUT";
        case ErrorCode::IncompatibleTemplate: return "INCOMPATIBLE_TEMPLATE";
        case ErrorCode::TensorMismatch: return "TENSOR_MISMATCH";
        case ErrorCode::OutputExists: return "OUTPUT_EXISTS";
        case ErrorCode::StaleOutput: return "STALE_OUTPUT";
        case ErrorCode::Busy: return "BUSY";
        case ErrorCode::ReadFailed: return "READ_FAILED";
        case ErrorCode::WriteFailed: return "WRITE_FAILED";
        case ErrorCode::InsufficientSpace: return "INSUFFICIENT_SPACE";
        case ErrorCode::NonFiniteWeight: return "NON_FINITE_WEIGHT";
        case ErrorCode::OutOfMemory: return "OUT_OF_MEMORY";
        case ErrorCode::CallbackFailed: return "CALLBACK_FAILED";
        case ErrorCode::CleanupFailed: return "CLEANUP_FAILED";
        case ErrorCode::InternalError: return "INTERNAL_ERROR";
    }
    return "INTERNAL_ERROR";
}

bool convert(const ConvertOptions& options, std::string& error) {
    const CancellationToken cancellation;
    const auto result = convert(options, {}, cancellation);
    error = result.error.message;
    if (result.status != ConvertStatus::Succeeded && error.empty()) {
        error = error_code_name(result.error.code);
    }
    return result.status == ConvertStatus::Succeeded;
}

}  // namespace sd15
