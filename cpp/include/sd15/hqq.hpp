#pragma once

#include <cstddef>
#include <cstdint>

namespace sd15 {

constexpr std::size_t kMaxHqqGroupElements = 65536;
constexpr int kHqqIterations = 20;
constexpr float kHqqLpNorm = 0.7f;
constexpr float kHqqBeta = 10.0f;
constexpr int kHqqBits = 8;
constexpr std::size_t kHqqAlphaValues = 2;
constexpr std::size_t kHqqAlphaBytes = kHqqAlphaValues * sizeof(float);

/// 한 그룹의 HQQ W8 인덱스와 FP32 min/scale을 계산한다.
void quantize_hqq(const float* weights, std::size_t count, uint8_t* payload, float* alpha,
                  int iterations = kHqqIterations);

}  // namespace sd15
