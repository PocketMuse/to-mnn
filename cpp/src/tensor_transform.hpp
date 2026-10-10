#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace sd15::detail {

enum class FloatStorage { F16, F32 };

constexpr std::size_t storage_width(FloatStorage storage) {
    return storage == FloatStorage::F16 ? 2 : 4;
}

// 입력·출력은 겹치지 않으며 count개 원소의 공간을 호출자가 제공한다.
// 실패하면 첫 비유한 원소의 인덱스를 반환한다. 출력은 일부만 채워질 수 있다.
std::optional<std::size_t> encode_float_chunk(
    const uint8_t* input, std::size_t count, FloatStorage source,
    FloatStorage target, bool positive_zero, uint8_t* output);

std::optional<std::size_t> decode_float_chunk(
    const uint8_t* input, std::size_t count, FloatStorage source,
    bool positive_zero, float* output);

// 유한성이 검증된 한 그룹을 W8 payload와 little-endian FP32 min/scale로 변환한다.
// 출력은 각각 count바이트와 kHqqAlphaBytes다. HQQ 범위 오류는 예외로 전달한다.
bool encode_hqq_group(const float* input, std::size_t count, int iterations,
                      uint8_t* payload, uint8_t* alpha);

}  // namespace sd15::detail
