#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sd15 {

constexpr int kSequenceLength = 77;
constexpr int kVocabularySize = 49408;
constexpr int kLatentElements = 4 * 64 * 64;
constexpr int kImageSize = 512;

struct SchedulerConfig {
    int train_steps = 1000;
    double beta_start = 0.00085;
    double beta_end = 0.012;
    int steps_offset = 1;
    bool set_alpha_to_one = false;
};

struct GenerateOptions {
    std::string model_dir;
    std::vector<int32_t> tokens;
    std::vector<int32_t> negative_tokens;
    std::vector<float> initial_latent;
    SchedulerConfig scheduler;
    int steps = 20;
    float guidance = 7.5f;
    float vae_scaling_factor = 0.18215f;
};

struct Image {
    int width = kImageSize;
    int height = kImageSize;
    std::vector<uint8_t> rgb;
};

// 동기 실행. 콜백도 호출 스레드에서 실행하며 오류는 std::exception으로 전달한다.
// 준비된 CLIP token IDs와 NCHW 초기 latent를 입력받는다. OpenCL만 사용한다.
using InferenceProgress = std::function<void(const std::string& stage, int step, int total)>;
Image generate(const GenerateOptions& options, const InferenceProgress& progress = {});

} // namespace sd15
