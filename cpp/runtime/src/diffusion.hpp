#pragma once

#include "sd15/runtime.hpp"

namespace sd15::detail {

struct SchedulerConfig {
    int train_steps = 1000;
    double beta_start = 0.00085;
    double beta_end = 0.012;
    int steps_offset = 1;
    bool set_alpha_to_one = false;
};

void validate(const GenerateOptions& options);
uint32_t resolve_seed(int64_t seed);
std::vector<float> initial_noise(uint32_t seed);
std::vector<float> alphas(const SchedulerConfig& config);
std::vector<int> timesteps(const SchedulerConfig& config, int steps);
void ddim_step(std::vector<float>& latent, const std::vector<float>& unconditional,
               const std::vector<float>& conditional, float guidance, float alpha, float previous_alpha);
Image pixels(const std::vector<float>& decoded);

} // namespace sd15::detail
