#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <exception>
#include <vector>

namespace sd15 {

constexpr int kSequenceLength = 77;
constexpr int kVocabularySize = 49408;
constexpr int kLatentElements = 4 * 64 * 64;
constexpr int kImageSize = 512;

struct GenerateOptions {
    std::string prompt;
    std::string negative_prompt;
    // 음수는 생성 요청마다 임의의 uint32 seed를 선택한다.
    int64_t seed = -1;
    int steps = 20;
    float guidance = 7.5f;
    int preview_interval = 0; // 0은 끔. 마지막 스텝도 간격에 맞으면 전달한다.
};

enum class Precision { High, Normal, Low };
enum class Tuning { None, Fast, Normal, Wide };

struct ModelOptions {
    Precision precision = Precision::High;
    Tuning tuning = Tuning::None;
    // MNN_LOW_MEMORY 빌드가 필요하다. 정수 연산을 보장하지 않는다.
    bool low_memory = false;
};

struct RuntimeOptions {
    std::string model_dir;
    // 비어 있으면 model_dir/tokenizer의 vocab.json·merges.txt를 사용한다.
    std::string tokenizer_dir;
    ModelOptions clip;
    ModelOptions unet;
    ModelOptions vae;
    std::string preview_decoder_path;
    ModelOptions preview;
};

struct Image {
    uint32_t seed = 0; // 실제 생성에 사용한 seed.
    int width = kImageSize;
    int height = kImageSize;
    std::vector<uint8_t> rgb;
};

// 동기 실행. 콜백도 호출 스레드에서 실행하며 오류는 예외로 전달한다.
// prompt는 UTF-8. batch 1, DDIM eta=0, OpenCL만 사용한다.
using InferenceProgress = std::function<void(const std::string& stage, int step, int total)>;
// 호출 스레드에서 실행. 이미지를 보관하려면 콜백 안에서 복사한다.
using InferencePreview = std::function<void(int step, int total, const Image& image)>;

class GenerationCancelled : public std::exception {
public:
    const char* what() const noexcept override { return "Generation cancelled"; }
};

// 요청마다 새 토큰을 사용하고 generate가 끝날 때까지 유지한다.
class GenerationCancellationToken {
public:
    void request_cancel() noexcept { requested_.store(true, std::memory_order_relaxed); }
    bool is_requested() const noexcept { return requested_.load(std::memory_order_relaxed); }

private:
    std::atomic<bool> requested_{false};
};

class Sd15Runtime {
public:
    explicit Sd15Runtime(const RuntimeOptions& options);
    ~Sd15Runtime();
    Sd15Runtime(const Sd15Runtime&) = delete;
    Sd15Runtime& operator=(const Sd15Runtime&) = delete;

    // 모델은 필요한 단계에만 로딩한다. 동시·콜백 내 재진입 호출은 거부한다.
    Image generate(const GenerateOptions& options, const InferenceProgress& progress = {},
                   const InferencePreview& preview = {});
    Image generate(const GenerateOptions& options, GenerationCancellationToken& cancellation,
                   const InferenceProgress& progress = {}, const InferencePreview& preview = {});

private:
    Image generate_impl(const GenerateOptions& options, const InferenceProgress& progress,
                        const GenerationCancellationToken* cancellation, const InferencePreview& preview);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sd15
