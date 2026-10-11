#pragma once

#include "sd15/runtime.hpp"

namespace sd15::detail {

void validate(const GenerateOptions& options);
std::vector<float> alphas(const SchedulerConfig& config);
std::vector<int> timesteps(const SchedulerConfig& config, int steps);
void ddim_step(std::vector<float>& latent, const std::vector<float>& unconditional,
               const std::vector<float>& conditional, float guidance, float alpha, float previous_alpha);
Image pixels(const std::vector<float>& decoded);

} // namespace sd15::detail
