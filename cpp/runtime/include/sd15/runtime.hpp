#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sd15 {

constexpr int kSequenceLength = 77;
constexpr int kVocabularySize = 49408;
constexpr int kLatentElements = 4 * 64 * 64;
constexpr int kImageSize = 512;

struct GenerateOptions {
    std::string prompt;
    std::string negative_prompt;
    uint32_t seed = 42;
    int steps = 20;
    float guidance = 7.5f;
};

struct RuntimeOptions {
    std::string model_dir;
    // 비어 있으면 model_dir/tokenizer의 vocab.json·merges.txt를 사용한다.
    std::string tokenizer_dir;
};

struct Image {
    int width = kImageSize;
    int height = kImageSize;
    std::vector<uint8_t> rgb;
};

// 동기 실행. 콜백도 호출 스레드에서 실행하며 오류는 예외로 전달한다.
// prompt는 UTF-8. batch 1, DDIM eta=0, OpenCL만 사용한다.
using InferenceProgress = std::function<void(const std::string& stage, int step, int total)>;

class Sd15Runtime {
public:
    explicit Sd15Runtime(const RuntimeOptions& options);
    ~Sd15Runtime();
    Sd15Runtime(const Sd15Runtime&) = delete;
    Sd15Runtime& operator=(const Sd15Runtime&) = delete;

    // 모델은 필요한 단계에만 로딩한다. 동시·콜백 내 재진입 호출은 거부한다.
    Image generate(const GenerateOptions& options, const InferenceProgress& progress = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sd15
