#include "diffusion.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace sd15::detail {

void validate_steps(int steps) {
    if (steps < 1 || steps >= kTrainSteps) {
        throw std::invalid_argument("steps must be in [1, train_steps)");
    }
}

void validate(const GenerateOptions& options) {
    if (options.preview_interval < 0) {
        throw std::invalid_argument("Preview interval must be nonnegative");
    }
    if (options.seed > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("seed must be negative or in [0, 4294967295]");
    }
    validate_steps(options.steps);
    if (!std::isfinite(options.guidance)) {
        throw std::invalid_argument("Guidance must be finite");
    }
}

uint32_t resolve_seed(int64_t seed) {
    if (seed > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("seed must be negative or in [0, 4294967295]");
    }
    if (seed >= 0) {
        return static_cast<uint32_t>(seed);
    }
    std::random_device source;
    std::uniform_int_distribution<uint32_t> distribution;
    return distribution(source);
}

std::vector<float> initial_noise(uint32_t seed) {
    // MNN diffusion 예제와 동일한 방식. 분포의 비트 재현성은 표준 라이브러리에 의존한다.
    std::mt19937 engine(seed);
    std::normal_distribution<float> normal(0, 1);
    std::vector<float> latent(kLatentElements);
    for (auto& value : latent) {
        value = normal(engine);
    }
    return latent;
}

void apply_guidance(std::vector<float>& unconditional, const std::vector<float>& conditional, float guidance) {
    assert(unconditional.size() == conditional.size());
    if (!std::isfinite(guidance)) {
        throw std::invalid_argument("Guidance must be finite");
    }
    for (std::size_t i = 0; i < unconditional.size(); ++i) {
        unconditional[i] = unconditional[i] + guidance * (conditional[i] - unconditional[i]);
    }
}

Image pixels(const std::vector<float>& decoded) {
    constexpr std::size_t plane = kImageSize * kImageSize;
    // VAE·TAESD의 출력 shape는 모델 초기화에서 확인한다.
    assert(decoded.size() == 3 * plane);
    Image image;
    image.rgb.resize(decoded.size());
    for (std::size_t i = 0; i < plane; ++i) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const float value = decoded[channel * plane + i];
            if (!std::isfinite(value)) {
                throw std::runtime_error("Non-finite decoded image");
            }
            const float scaled = std::clamp(value / 2 + 0.5f, 0.0f, 1.0f) * 255;
            image.rgb[3 * i + channel] = static_cast<uint8_t>(std::round(scaled));
        }
    }
    return image;
}

} // namespace sd15::detail
