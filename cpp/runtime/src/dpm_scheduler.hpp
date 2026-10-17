#pragma once

#include "scheduler.hpp"

namespace sd15::detail {

// Diffusers 0.40.0: dpmsolver++, order=2, midpoint, Karras, epsilon, final sigma=0.
class DpmScheduler final : public Scheduler {
public:
    explicit DpmScheduler(int steps);
    const std::vector<int>& timesteps() const noexcept override { return timesteps_; }
    const std::vector<float>& sigmas() const noexcept { return sigmas_; }
    float init_noise_sigma() const noexcept override { return 1.0f; }
    const std::vector<float>& scale_model_input(const std::vector<float>& sample, std::size_t) const override {
        return sample;
    }
    void step(std::vector<float>& latent, const std::vector<float>& model_output, std::size_t index,
              std::vector<float>* clean_prediction = nullptr) override;

private:
    std::vector<int> timesteps_;
    std::vector<float> sigmas_;
    std::vector<float> previous_clean_;
    std::size_t next_index_ = 0;
};

} // namespace sd15::detail
