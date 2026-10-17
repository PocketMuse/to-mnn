#pragma once

#include "sd15/runtime.hpp"

#include <cstddef>
#include <memory>

namespace sd15::detail {

// 생성 요청이 소유하는 scheduler 전략. 모델 실행과 CFG는 Runtime이 담당한다.
class Scheduler {
public:
    virtual ~Scheduler() = default;
    virtual const std::vector<int>& timesteps() const noexcept = 0;
    virtual float init_noise_sigma() const noexcept = 0;
    virtual const std::vector<float>& scale_model_input(const std::vector<float>& sample, std::size_t index) const = 0;
    virtual void step(std::vector<float>& latent, const std::vector<float>& model_output, std::size_t index,
                      std::vector<float>* clean_prediction = nullptr) = 0;
};

std::unique_ptr<Scheduler> make_scheduler(SamplingPreset preset, int steps);

} // namespace sd15::detail
