#pragma once

#include "sd15/runtime.hpp"

namespace sd15::detail {

inline constexpr int kTrainSteps = 1000;

void validate_steps(int steps);
void validate(const GenerateOptions& options);
uint32_t resolve_seed(int64_t seed);
std::vector<float> initial_noise(uint32_t seed);
// unconditional 버퍼를 CFG 결과로 갱신한다.
void apply_guidance(std::vector<float>& unconditional, const std::vector<float>& conditional, float guidance);
// 모델 초기화에서 확인한 [1, 3, 512, 512] NCHW 출력만 받는다.
Image pixels(const std::vector<float>& decoded);

} // namespace sd15::detail
