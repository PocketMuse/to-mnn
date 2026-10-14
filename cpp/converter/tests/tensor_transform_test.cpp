#include "tensor_transform.hpp"
#include "sd15/hqq.hpp"

#include <array>
#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>

using namespace sd15::detail;

TEST_CASE("Float chunks preserve bytes and normalize zero") {
    // 1, -0, 최소 FP16 subnormal. 앞뒤 표식으로 출력 범위도 검사한다.
    const std::array<uint8_t, 6> source{0x00, 0x3c, 0x00, 0x80, 0x01, 0x00};
    const std::array<uint8_t, 12> expected{0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x80, 0x33};
    std::array<uint8_t, 14> output;
    output.fill(0xa5);
    UNSCOPED_INFO("half expansion");
    REQUIRE(!encode_float_chunk(source.data(), 3, FloatStorage::F16, FloatStorage::F32, false, output.data() + 1));
    UNSCOPED_INFO("output bounds");
    REQUIRE(output.front() == 0xa5);
    REQUIRE(output.back() == 0xa5);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        UNSCOPED_INFO("exact float bytes");
        REQUIRE(output[i + 1] == expected[i]);
    }

    std::array<uint8_t, 6> restored{};
    UNSCOPED_INFO("half encoding");
    REQUIRE(!encode_float_chunk(expected.data(), 3, FloatStorage::F32, FloatStorage::F16, false, restored.data()));
    UNSCOPED_INFO("half round trip");
    REQUIRE(restored == source);
    UNSCOPED_INFO("zero normalization");
    REQUIRE(!encode_float_chunk(expected.data(), 3, FloatStorage::F32, FloatStorage::F16, true, restored.data()));
    UNSCOPED_INFO("source negative zero normalized");
    REQUIRE(restored[3] == 0);

    std::array<float, 3> values{};
    UNSCOPED_INFO("decode half weights");
    REQUIRE(!decode_float_chunk(source.data(), 3, FloatStorage::F16, false, values.data()));
    UNSCOPED_INFO("decoded half values");
    REQUIRE(values[0] == 1);
    REQUIRE(std::signbit(values[1]));
    REQUIRE(values[2] == std::ldexp(1.0f, -24));
    UNSCOPED_INFO("decode float weights");
    REQUIRE(!decode_float_chunk(expected.data(), 3, FloatStorage::F32, true, values.data()));
    UNSCOPED_INFO("decoded zero normalized");
    REQUIRE(!std::signbit(values[1]));
    UNSCOPED_INFO("empty encoding");
    REQUIRE(!encode_float_chunk(nullptr, 0, FloatStorage::F32, FloatStorage::F16, false, nullptr));
    UNSCOPED_INFO("empty decoding");
    REQUIRE(!decode_float_chunk(nullptr, 0, FloatStorage::F32, false, nullptr));
}

TEST_CASE("Nonfinite values report their index") {
    const std::array<uint8_t, 4> half{0x00, 0x3c, 0x00, 0x7c};
    const std::array<uint8_t, 8> single{0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0xc0, 0x7f};
    std::array<uint8_t, 8> output{};
    std::array<float, 2> values{};
    for (const auto storage : {FloatStorage::F16, FloatStorage::F32}) {
        const auto* input = storage == FloatStorage::F16 ? half.data() : single.data();
        UNSCOPED_INFO("encoding failure index");
        REQUIRE(encode_float_chunk(input, 2, storage, FloatStorage::F32, false, output.data()) == 1);
        UNSCOPED_INFO("decoding failure index");
        REQUIRE(decode_float_chunk(input, 2, storage, false, values.data()) == 1);
        UNSCOPED_INFO("first element failure");
        REQUIRE(decode_float_chunk(input + storage_width(storage), 1, storage, false, values.data()) == 0);
    }
}

TEST_CASE("HQQ storage matches expected payload and alpha") {
    const std::array<float, 4> input{0, 0.49f, 1.49f, 255};
    std::array<uint8_t, 4> payload{};
    std::array<uint8_t, sd15::kHqqAlphaBytes> alpha{};
    const std::array<uint8_t, 4> expected_payload{0, 0, 1, 255};
    const std::array<uint8_t, sd15::kHqqAlphaBytes> expected_alpha{0, 0, 0, 0, 0, 0, 0x80, 0x3f};
    UNSCOPED_INFO("HQQ encoding");
    REQUIRE(encode_hqq_group(input.data(), input.size(), 0, payload.data(), alpha.data()) == sd15::HqqError::None);
    UNSCOPED_INFO("HQQ storage bytes");
    REQUIRE(payload == expected_payload);
    REQUIRE(alpha == expected_alpha);
}

TEST_CASE("HQQ classifies invalid groups and overflow") {
    uint8_t payload[2]{};
    float alpha[sd15::kHqqAlphaValues]{};
    const float maximum = std::numeric_limits<float>::max();
    const float range[2]{-maximum, maximum};
    const float normalization[2]{maximum, maximum};
    UNSCOPED_INFO("invalid group result");
    REQUIRE(sd15::quantize_hqq(range, 0, payload, alpha) == sd15::HqqError::InvalidGroupSize);
    UNSCOPED_INFO("invalid iterations result");
    REQUIRE(sd15::quantize_hqq(range, 2, payload, alpha, -1) == sd15::HqqError::InvalidIterations);
    UNSCOPED_INFO("range overflow result");
    REQUIRE(sd15::quantize_hqq(range, 2, payload, alpha) == sd15::HqqError::RangeOverflow);
    UNSCOPED_INFO("normalization overflow result");
    REQUIRE(sd15::quantize_hqq(normalization, 2, payload, alpha) == sd15::HqqError::NormalizationOverflow);
}
