#include "dpm_scheduler.hpp"
#include "diffusion.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace sd15::detail {
namespace {

constexpr double kKarrasRho = 7.0;

std::vector<double> training_sigmas() {
    constexpr double kBetaStart = 0.00085;
    constexpr double kBetaEnd = 0.012;
    const double first = std::sqrt(kBetaStart);
    const double increment = (std::sqrt(kBetaEnd) - first) / (kTrainSteps - 1);
    double alpha = 1.0;
    std::vector<double> result;
    result.reserve(kTrainSteps);
    // 학습 beta·alpha는 FP32, 누적곱과 보간은 double로 계산한다.
    for (int i = 0; i < kTrainSteps; ++i) {
        const double root = first + increment * i;
        const float beta = static_cast<float>(root * root);
        alpha *= 1.0f - beta;
        const float stored_alpha = static_cast<float>(alpha);
        result.push_back(std::sqrt((1.0 - stored_alpha) / stored_alpha));
    }
    return result;
}

double sigma_timestep(double sigma, const std::vector<double>& log_sigmas) {
    const double value = std::log(sigma);
    const auto upper = std::upper_bound(log_sigmas.begin(), log_sigmas.end(), value);
    const auto low = std::clamp<std::ptrdiff_t>(upper - log_sigmas.begin() - 1, 0, log_sigmas.size() - 2);
    const double fraction = std::clamp((value - log_sigmas[low]) / (log_sigmas[low + 1] - log_sigmas[low]), 0.0, 1.0);
    return low + fraction;
}

int round_timestep(double time) {
    const int floor = static_cast<int>(std::floor(time));
    const double remainder = time - floor;
    // NumPy round와 같은 ties-to-even. 실행 스레드의 반올림 모드에 의존하지 않는다.
    return floor + (remainder > 0.5 || (remainder == 0.5 && floor % 2 != 0) ? 1 : 0);
}

struct NoiseLevel {
    explicit NoiseLevel(float ratio) : alpha(1.0f / std::sqrt(ratio * ratio + 1.0f)), sigma(ratio * alpha) {}
    float lambda() const { return std::log(alpha) - std::log(sigma); }
    float alpha;
    float sigma;
};

} // namespace

DpmScheduler::DpmScheduler(int steps) {
    validate_steps(steps);
    const auto trained = training_sigmas();
    std::vector<double> log_sigmas;
    log_sigmas.reserve(trained.size());
    for (double sigma : trained) {
        log_sigmas.push_back(std::log(sigma));
    }
    const double high = std::pow(trained.back(), 1.0 / kKarrasRho);
    const double low = std::pow(trained.front(), 1.0 / kKarrasRho);
    for (int i = 0; i < steps; ++i) {
        const double fraction = steps == 1 ? 0.0 : static_cast<double>(i) / (steps - 1);
        const double sigma = std::pow(high + fraction * (low - high), kKarrasRho);
        sigmas_.push_back(static_cast<float>(sigma));
        const double time = sigma_timestep(sigma, log_sigmas);
        timesteps_.push_back(round_timestep(time));
    }
    sigmas_.push_back(0.0f);
}

void DpmScheduler::step(std::vector<float>& latent, const std::vector<float>& model_output, std::size_t index,
                        std::vector<float>* clean_prediction) {
    if (index != next_index_ || index >= timesteps_.size()) {
        throw std::invalid_argument("DPM steps must be executed in order");
    }
    assert(latent.size() == model_output.size());
    assert(index == 0 || previous_clean_.size() == latent.size());
    const NoiseLevel current(sigmas_[index]);
    const NoiseLevel next(sigmas_[index + 1]);
    const bool final = index + 1 == timesteps_.size();
    const bool second_order = index > 0 && !final;
    // sigma=0의 마지막 스텝은 1차 갱신의 극한인 x0를 반환한다.
    const float h = final ? 0.0f : next.lambda() - current.lambda();
    const float coefficient = final ? -1.0f : next.alpha * (std::exp(-h) - 1.0f);
    const float history_ratio = second_order ? h / (current.lambda() - NoiseLevel(sigmas_[index - 1]).lambda()) : 0.0f;
    if (index == 0) {
        previous_clean_.resize(latent.size());
    }
    if (clean_prediction != nullptr) {
        clean_prediction->resize(latent.size());
    }
    for (std::size_t i = 0; i < latent.size(); ++i) {
        const float clean = (latent[i] - current.sigma * model_output[i]) / current.alpha;
        float value = final ? clean : (next.sigma / current.sigma) * latent[i] - coefficient * clean;
        if (second_order) {
            const float derivative = history_ratio * (clean - previous_clean_[i]);
            value -= 0.5f * coefficient * derivative;
        }
        if (!std::isfinite(clean) || !std::isfinite(value)) {
            throw std::runtime_error("Non-finite DPM latent");
        }
        previous_clean_[i] = clean;
        latent[i] = value;
        if (clean_prediction != nullptr) {
            (*clean_prediction)[i] = clean;
        }
    }
    ++next_index_;
}

} // namespace sd15::detail
