#include "sd15/hqq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sd15 {

void quantize_hqq(const float* weights, std::size_t count, uint8_t* payload, float* alpha) {
    if (count == 0 || count > kMaxHqqGroupElements) {
        throw std::runtime_error("Invalid HQQ group size");
    }
    constexpr float kQuantRange = 255.0f;
    constexpr float kMinScale = 1e-7f;
    constexpr float kMinPowInput = 1e-8f;
    constexpr float kQuantOffset = 128.0f;
    float minimum = std::numeric_limits<float>::max();
    float maximum = -std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < count; ++i) {
        minimum = std::min(minimum, weights[i]);
        maximum = std::max(maximum, weights[i]);
    }
    const float initial_scale = std::max((maximum - minimum) * (1.0f / kQuantRange), kMinScale);
    if (!std::isfinite(initial_scale)) {
        throw std::runtime_error("HQQ range overflow");
    }
    const float inverse = 1.0f / initial_scale;
    const float scale = 1.0f / inverse;
    float zero = inverse * -minimum;
    if (!std::isfinite(zero) || !std::isfinite(maximum * inverse)) {
        throw std::runtime_error("HQQ normalization overflow");
    }
    // MNN 3.6.1은 scale을 고정하고 같은 beta로 zero만 20회 갱신한다.
    for (int iteration = 0; iteration < kHqqIterations; ++iteration) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < count; ++i) {
            const float rounded = std::clamp(std::round(weights[i] * inverse + zero), 0.0f, kQuantRange);
            const float restored = (rounded - zero) * scale;
            const float difference = weights[i] - restored;
            const float magnitude = std::abs(difference);
            const float power = std::pow(std::max(magnitude, kMinPowInput), kHqqLpNorm - 1.0f);
            const float shrunk = std::max(magnitude - (1.0f / kHqqBeta) * power, 0.0f);
            const float sign = difference > 0.0f ? 1.0f : (difference < 0.0f ? -1.0f : 0.0f);
            const float error = shrunk * sign;
            sum += rounded - (weights[i] - error) * inverse;
        }
        zero = sum / static_cast<float>(count);
        if (!std::isfinite(zero)) {
            throw std::runtime_error("HQQ zero overflow");
        }
    }
    minimum = -(zero * scale);
    if (!std::isfinite(minimum)) {
        throw std::runtime_error("HQQ minimum overflow");
    }
    alpha[0] = minimum;
    alpha[1] = scale;
    for (std::size_t i = 0; i < count; ++i) {
        const float value = (weights[i] - minimum) * inverse - kQuantOffset;
        const float quantized = std::clamp(std::round(value), -kQuantOffset, kQuantOffset - 1.0f);
        payload[i] = static_cast<uint8_t>(quantized + kQuantOffset);
    }
}

}  // namespace sd15
