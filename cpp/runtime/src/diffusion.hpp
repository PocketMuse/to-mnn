#pragma once

#include "sd15/runtime.hpp"

namespace sd15::detail {

inline constexpr int kTrainSteps = 1000;

void validate(const GenerateOptions& options);
uint32_t resolve_seed(int64_t seed);
std::vector<float> initial_noise(uint32_t seed);
std::vector<float> alphas();
std::vector<int> timesteps(int steps);
// 세 버퍼의 크기는 같아야 한다. latent 크기는 변경하지 않는다.
void ddim_step(std::vector<float>& latent, const std::vector<float>& unconditional,
               const std::vector<float>& conditional, float guidance, float alpha, float previous_alpha,
               std::vector<float>* clean_prediction = nullptr);
// 모델 초기화에서 확인한 [1, 3, 512, 512] NCHW 출력만 받는다.
Image pixels(const std::vector<float>& decoded);

} // namespace sd15::detail
