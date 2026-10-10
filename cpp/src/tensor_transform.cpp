#include "tensor_transform.hpp"
#include "sd15/hqq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace sd15::detail {
namespace {

constexpr uint32_t kBitsPerByte = 8;
constexpr uint32_t kHalfMaxFloatBits = 0x477fe000;
constexpr uint32_t kHalfSign = 0x8000;
constexpr uint32_t kHalfSignShift = 16;
constexpr uint32_t kHalfMantissa = 0x3ff;
constexpr uint32_t kHalfImplicitBit = 0x400;
constexpr uint32_t kHalfExponentMask = 0x1f;
constexpr uint32_t kFloatExponentMask = 0xff;
constexpr uint32_t kFloatSign = 0x80000000;
constexpr uint32_t kFloatMagnitude = 0x7fffffff;
constexpr uint32_t kFloatMantissa = 0x007fffff;
constexpr uint32_t kFloatImplicitBit = 0x00800000;
constexpr uint32_t kFloatExponentBits = 23;
constexpr uint32_t kHalfExponentBits = 10;
constexpr uint32_t kExponentBiasDifference = 112;
constexpr uint32_t kFirstHalfSubnormalExponent = 103;
constexpr uint32_t kMantissaShift = 13;
constexpr uint32_t kSubnormalShiftBias = 126;

uint64_t read_le(const uint8_t* bytes, std::size_t size) {
    uint64_t value = 0;
    for (std::size_t i = 0; i < size; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (i * kBitsPerByte);
    }
    return value;
}

void write_le(uint8_t* bytes, uint32_t value, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<uint8_t>(value >> (i * kBitsPerByte));
    }
}

/// subnormal과 signed zero를 포함해 FP16을 FP32 비트로 확장한다.
uint32_t half_to_float_bits(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & kHalfSign) << kHalfSignShift;
    uint32_t exponent = (half >> kHalfExponentBits) & kHalfExponentMask;
    uint32_t mantissa = half & kHalfMantissa;
    if (exponent == 0 && mantissa == 0) {
        return sign;
    }
    if (exponent == 0) {
        int shift = 0;
        while ((mantissa & kHalfImplicitBit) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        exponent = kExponentBiasDifference + 1 - shift;
        return sign | (exponent << kFloatExponentBits) | ((mantissa & kHalfMantissa) << kMantissaShift);
    }
    exponent = exponent == kHalfExponentMask ? kFloatExponentMask : exponent + kExponentBiasDifference;
    return sign | (exponent << kFloatExponentBits) | (mantissa << kMantissaShift);
}

/// 유한 FP32를 MNN 규칙에 따라 범위 제한 후 0 방향으로 FP16 절삭한다.
uint16_t float_to_half_bits(uint32_t bits) {
    const uint16_t sign = static_cast<uint16_t>((bits & kFloatSign) >> kHalfSignShift);
    const uint32_t magnitude = std::min(bits & kFloatMagnitude, kHalfMaxFloatBits);
    const uint32_t exponent = magnitude >> kFloatExponentBits;
    if (exponent < kFirstHalfSubnormalExponent) {
        return sign;
    }
    if (exponent <= kExponentBiasDifference) {
        return sign | static_cast<uint16_t>(((magnitude & kFloatMantissa) | kFloatImplicitBit) >> (kSubnormalShiftBias - exponent));
    }
    return sign | static_cast<uint16_t>(((exponent - kExponentBiasDifference) << kHalfExponentBits) |
                                       ((magnitude & kFloatMantissa) >> kMantissaShift));
}

// 원본 signed zero만 정규화한다. FP16 절삭으로 생기는 -0은 보존한다.
bool decode_bits(const uint8_t* input, FloatStorage storage, bool positive_zero, uint32_t& bits) {
    bits = static_cast<uint32_t>(read_le(input, storage_width(storage)));
    if (storage == FloatStorage::F16) {
        bits = half_to_float_bits(static_cast<uint16_t>(bits));
    }
    if (((bits >> kFloatExponentBits) & kFloatExponentMask) == kFloatExponentMask) {
        return false;
    }
    if (positive_zero && (bits & kFloatMagnitude) == 0) {
        bits = 0;
    }
    return true;
}

}  // namespace

std::optional<std::size_t> encode_float_chunk(
    const uint8_t* input, std::size_t count, FloatStorage source,
    FloatStorage target, bool positive_zero, uint8_t* output) {
    const auto input_width = storage_width(source);
    const auto output_width = storage_width(target);
    for (std::size_t i = 0; i < count; ++i) {
        uint32_t bits;
        if (!decode_bits(input + i * input_width, source, positive_zero, bits)) {
            return i;
        }
        const uint32_t value = target == FloatStorage::F16 ? float_to_half_bits(bits) : bits;
        write_le(output + i * output_width, value, output_width);
    }
    return std::nullopt;
}

std::optional<std::size_t> decode_float_chunk(
    const uint8_t* input, std::size_t count, FloatStorage source,
    bool positive_zero, float* output) {
    const auto width = storage_width(source);
    for (std::size_t i = 0; i < count; ++i) {
        uint32_t bits;
        if (!decode_bits(input + i * width, source, positive_zero, bits)) {
            return i;
        }
        std::memcpy(output + i, &bits, sizeof(bits));
    }
    return std::nullopt;
}

HqqError encode_hqq_group(const float* input, std::size_t count, int iterations,
                          uint8_t* payload, uint8_t* alpha) noexcept {
    float pair[kHqqAlphaValues];
    const auto error = quantize_hqq(input, count, payload, pair, iterations);
    if (error != HqqError::None) {
        return error;
    }
    for (std::size_t i = 0; i < kHqqAlphaValues; ++i) {
        if (!std::isfinite(pair[i])) {
            return HqqError::NonFiniteScale;
        }
        uint32_t bits;
        std::memcpy(&bits, &pair[i], sizeof(bits));
        write_le(alpha + i * sizeof(float), bits, sizeof(bits));
    }
    return HqqError::None;
}

}  // namespace sd15::detail
