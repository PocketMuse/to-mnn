#include "ddim_scheduler.hpp"
#include "diffusion.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace sd15::detail {
namespace {
constexpr double kBetaStart = 0.00085;
constexpr double kBetaEnd = 0.012;
constexpr int kStepsOffset = 1;
static_assert(kTrainSteps > 1 && kStepsOffset == 1);
static_assert(kBetaStart > 0 && kBetaStart <= kBetaEnd && kBetaEnd < 1);

std::vector<int> make_timesteps(int steps) {
    validate_steps(steps);
    const int stride = kTrainSteps / steps;
    std::vector<int> values;
    for (int i = steps - 1; i >= 0; --i) {
        const int time = i * stride + kStepsOffset;
        values.push_back(time);
    }
    return values;
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

void ddim_step(std::vector<float>& latent, const std::vector<float>& model_output, float alpha, float previous_alpha,
               std::vector<float>* clean_prediction) {
    // latent는 크기를 유지하고 noise는 초기화 때 연결한 UNet 출력이다.
    assert(latent.size() == model_output.size());
    if (!(alpha > 0 && alpha <= 1 && previous_alpha > 0 && previous_alpha <= 1)) {
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
        const float noise = model_output[i];
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

} // namespace

DdimScheduler::DdimScheduler(int steps) : timesteps_(make_timesteps(steps)), alphas_(alphas()) {
}

void DdimScheduler::step(std::vector<float>& latent, const std::vector<float>& model_output, std::size_t index,
                         std::vector<float>* clean_prediction) {
    assert(index < timesteps_.size());
    const int time = timesteps_[index];
    const int previous = time - kTrainSteps / static_cast<int>(timesteps_.size());
    // 기존 SD1.5 계약: leading, offset=1, eta=0, set_alpha_to_one=false.
    const float previous_alpha = previous >= 0 ? alphas_[previous] : alphas_.front();
    ddim_step(latent, model_output, alphas_[time], previous_alpha, clean_prediction);
}

} // namespace sd15::detail
