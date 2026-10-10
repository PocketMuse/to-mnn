#pragma once

#include "sd15/hqq.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace sd15 {

constexpr std::size_t kDefaultChunkBytes = 1024 * 1024;

/// 입력 체크포인트·템플릿 경로와 새 출력 디렉터리, 작업 청크 크기.
struct ConvertOptions {
    std::string checkpoint;
    std::string template_dir;
    std::string output_dir;
    /// 입력 버퍼 크기. 4바이트~4MiB의 4의 배수이며 출력 버퍼는 두 배다.
    std::size_t chunk_bytes = kDefaultChunkBytes;
    /// HQQ 보정 횟수 [0, 20]. 0은 초기 min/max로 W8 양자화한다.
    int hqq_iterations = kHqqIterations;
};

enum class ErrorCode {
    None, InvalidInput, IncompatibleTemplate, TensorMismatch, OutputExists,
    StaleOutput, Busy, ReadFailed, WriteFailed, InsufficientSpace,
    NonFiniteWeight, OutOfMemory, CallbackFailed, CleanupFailed, InternalError
};

struct ConvertError {
    ErrorCode code = ErrorCode::None;
    std::string message;
    std::string path;
    std::string tensor;
};

enum class ConvertStage { Validating, Converting, Finalizing, Completed };
enum class ConvertStatus { Succeeded, Cancelled, Failed };

struct OutputFile {
    std::string name;
    uint64_t size_bytes = 0;
};

struct InspectResult {
    bool ok = false;
    std::string model_family = "sd15";
    int manifest_version = 0;
    std::string mnn_version;
    /// manifest의 정책 식별자. 개별 텐서의 저장 dtype은 혼합될 수 있다.
    std::string unet_quantization;
    std::vector<OutputFile> files;
    uint64_t output_bytes = 0;
    std::optional<uint64_t> available_bytes;
    bool partial_exists = false;
    ConvertError error;
    /// 양자화 대상이 없으면 값 없음.
    std::optional<int> hqq_iterations;
};

struct ConvertProgress {
    ConvertStage stage = ConvertStage::Validating;
    std::string component;
    uint64_t completed_bytes = 0;
    uint64_t total_bytes = 0;
    /// 출력 공개 전에는 최대 0.99, 완료 후에만 1.0이다.
    double fraction = 0;
    uint64_t elapsed_ms = 0;
    std::optional<uint64_t> remaining_ms;
};

struct ConvertResult {
    ConvertStatus status = ConvertStatus::Failed;
    std::vector<OutputFile> files;
    uint64_t elapsed_ms = 0;
    ConvertError error;
    std::optional<int> hqq_iterations;
    /// HQQ 계산 배치의 경과 시간. 입력 변환과 파일 I/O는 제외한다.
    double hqq_compute_ms = 0;
};

/// 변환마다 새 토큰을 사용한다. 반복 요청은 안전하며 리셋하지 않는다.
class CancellationToken {
public:
    void request_cancel() noexcept { requested_.store(true, std::memory_order_relaxed); }
    bool is_requested() const noexcept { return requested_.load(std::memory_order_relaxed); }
private:
    std::atomic<bool> requested_{false};
};

using ProgressCallback = std::function<void(const ConvertProgress&)>;

/// 헤더·manifest만 검사한다. 가중치 본문의 유한성은 변환 중 검사한다.
InspectResult inspect(const ConvertOptions& options);

/// 호출 스레드에서 실행한다. 콜백도 같은 스레드에서 호출한다.
ConvertResult convert(const ConvertOptions& options, const ProgressCallback& on_progress,
                      const CancellationToken& cancellation);

struct CleanupResult {
    bool ok = false;
    ConvertError error;
};

/// 소유 표식이 있는 중단된 출력만 정리한다. 프로세스 내 변환 중에는 거부한다.
CleanupResult cleanup_partial(const ConvertOptions& options);

/// JS 연결 계층에서 사용할 안정적인 오류 코드 문자열.
const char* error_code_name(ErrorCode code) noexcept;

/// F32/F16 safetensors와 템플릿으로 FP16 또는 HQQ W8 MNN을 스트리밍 복원한다.
/// 성공 시 true, 실패 시 false와 error를 반환한다. 기존 출력은 거부한다.
/// 완성 전 결과는 .partial에 쓰며 일반 오류 시 이번 작업 파일만 정리한다.
bool convert(const ConvertOptions& options, std::string& error);

}  // namespace sd15
