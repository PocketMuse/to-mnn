#pragma once

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

// c2af64f의 DDIM 계산을 보존한 회귀 기준. 제품 코드에서는 사용하지 않는다.
namespace legacy {
constexpr int kTrainSteps = 1000;
constexpr double kBetaStart = 0.00085;
constexpr double kBetaEnd = 0.012;

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
    // latent는 크기를 유지하고 noise는 초기화 때 연결한 UNet 출력이다.
    assert(latent.size() == unconditional.size() && latent.size() == conditional.size());
    if (!(alpha > 0 && alpha <= 1 && previous_alpha > 0 && previous_alpha <= 1) || !std::isfinite(guidance)) {
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

}
