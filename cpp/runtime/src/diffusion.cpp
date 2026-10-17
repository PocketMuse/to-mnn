#include "diffusion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace sd15::detail {
namespace {

constexpr double kBetaStart = 0.00085;
constexpr double kBetaEnd = 0.012;
constexpr int kStepsOffset = 1;
static_assert(kTrainSteps > 1 && kStepsOffset == 1);
static_assert(kBetaStart > 0 && kBetaStart <= kBetaEnd && kBetaEnd < 1);

void validate_steps(int steps) {
    if (steps < 1 || steps >= kTrainSteps) {
        throw std::invalid_argument("steps must be in [1, train_steps)");
    }
}

} // namespace

std::vector<int> timesteps(int steps) {
    validate_steps(steps);
    const int stride = kTrainSteps / steps;
    std::vector<int> values;
    for (int i = steps - 1; i >= 0; --i) {
        const int time = i * stride + kStepsOffset;
        values.push_back(time);
    }
    return values;
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

std::vector<float> alphas() {
    const double first = std::sqrt(kBetaStart);
    const double step = (std::sqrt(kBetaEnd) - first) / (kTrainSteps - 1);
    float product = 1;
    std::vector<float> result;
    for (int i = 0; i < kTrainSteps; ++i) {
        const float root = static_cast<float>(first + i * step);
        product *= 1.0f - root * root;
        if (!std::isfinite(product) || product <= 0) {
            throw std::invalid_argument("Invalid cumulative alpha");
        }
        result.push_back(product);
    }
    return result;
}

void ddim_step(std::vector<float>& latent, const std::vector<float>& unconditional,
               const std::vector<float>& conditional, float guidance, float alpha, float previous_alpha,
               std::vector<float>* clean_prediction) {
    if (latent.size() != unconditional.size() || latent.size() != conditional.size() ||
        !(alpha > 0 && alpha <= 1 && previous_alpha > 0 && previous_alpha <= 1) || !std::isfinite(guidance)) {
        throw std::invalid_argument("Invalid DDIM inputs");
    }
    const float signal = std::sqrt(alpha);
    if (clean_prediction != nullptr) {
        clean_prediction->resize(latent.size());
    }
    const float noise_scale = std::sqrt(1 - alpha);
    const float previous_signal = std::sqrt(previous_alpha);
    const float previous_noise = std::sqrt(1 - previous_alpha);
    for (std::size_t i = 0; i < latent.size(); ++i) {
        const float noise = unconditional[i] + guidance * (conditional[i] - unconditional[i]);
        const float clean = (latent[i] - noise_scale * noise) / signal;
        if (clean_prediction != nullptr) {
            (*clean_prediction)[i] = clean;
        }
        latent[i] = previous_signal * clean + previous_noise * noise;
        if (!std::isfinite(latent[i])) {
            throw std::runtime_error("Non-finite DDIM latent");
        }
    }
}

Image pixels(const std::vector<float>& decoded) {
    constexpr std::size_t plane = kImageSize * kImageSize;
    if (decoded.size() != 3 * plane) {
        throw std::invalid_argument("Expected decoded shape [1, 3, 512, 512]");
    }
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
