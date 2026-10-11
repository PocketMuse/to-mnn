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

enum class HqqError {
    None,
    InvalidGroupSize,
    InvalidIterations,
    RangeOverflow,
    NormalizationOverflow,
    ZeroOverflow,
    MinimumOverflow,
    NonFiniteScale
};

/// 유한 입력 한 그룹을 변환한다. 실패 시 출력은 사용하지 않는다.
HqqError quantize_hqq(const float* weights, std::size_t count, uint8_t* payload, float* alpha,
                      int iterations = kHqqIterations) noexcept;
const char* hqq_error_message(HqqError error) noexcept;

} // namespace sd15
