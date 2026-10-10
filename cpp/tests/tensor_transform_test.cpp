#include "tensor_transform.hpp"
#include "sd15/hqq.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sd15::detail;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_float_chunks() {
    // 1, -0, 최소 FP16 subnormal. 앞뒤 표식으로 출력 범위도 검사한다.
    const std::array<uint8_t, 6> source{0x00, 0x3c, 0x00, 0x80, 0x01, 0x00};
    const std::array<uint8_t, 12> expected{
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x80, 0x33};
    std::array<uint8_t, 14> output;
    output.fill(0xa5);
    require(!encode_float_chunk(source.data(), 3, FloatStorage::F16, FloatStorage::F32,
                                false, output.data() + 1), "half expansion");
    require(output.front() == 0xa5 && output.back() == 0xa5, "output bounds");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(output[i + 1] == expected[i], "exact float bytes");
    }

    std::array<uint8_t, 6> restored{};
    require(!encode_float_chunk(expected.data(), 3, FloatStorage::F32, FloatStorage::F16,
                                false, restored.data()), "half encoding");
    require(restored == source, "half round trip");
    require(!encode_float_chunk(expected.data(), 3, FloatStorage::F32, FloatStorage::F16,
                                true, restored.data()), "zero normalization");
    require(restored[3] == 0, "source negative zero normalized");

    std::array<float, 3> values{};
    require(!decode_float_chunk(source.data(), 3, FloatStorage::F16, false, values.data()),
            "decode half weights");
    require(values[0] == 1 && std::signbit(values[1]) && values[2] == std::ldexp(1.0f, -24),
            "decoded half values");
    require(!decode_float_chunk(expected.data(), 3, FloatStorage::F32, true, values.data()),
            "decode float weights");
    require(!std::signbit(values[1]), "decoded zero normalized");
    require(!encode_float_chunk(nullptr, 0, FloatStorage::F32, FloatStorage::F16, false, nullptr),
            "empty encoding");
    require(!decode_float_chunk(nullptr, 0, FloatStorage::F32, false, nullptr), "empty decoding");
}

void check_nonfinite() {
    const std::array<uint8_t, 4> half{0x00, 0x3c, 0x00, 0x7c};
    const std::array<uint8_t, 8> single{0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0xc0, 0x7f};
    std::array<uint8_t, 8> output{};
    std::array<float, 2> values{};
    for (const auto storage : {FloatStorage::F16, FloatStorage::F32}) {
        const auto* input = storage == FloatStorage::F16 ? half.data() : single.data();
        require(encode_float_chunk(input, 2, storage, FloatStorage::F32, false, output.data()) == 1,
                "encoding failure index");
        require(decode_float_chunk(input, 2, storage, false, values.data()) == 1,
                "decoding failure index");
        require(decode_float_chunk(input + storage_width(storage), 1, storage, false, values.data()) == 0,
                "first element failure");
    }
}

void check_hqq_storage() {
    const std::array<float, 4> input{0, 0.49f, 1.49f, 255};
    std::array<uint8_t, 4> payload{};
    std::array<uint8_t, sd15::kHqqAlphaBytes> alpha{};
    const std::array<uint8_t, 4> expected_payload{0, 0, 1, 255};
    const std::array<uint8_t, sd15::kHqqAlphaBytes> expected_alpha{0, 0, 0, 0, 0, 0, 0x80, 0x3f};
    require(encode_hqq_group(input.data(), input.size(), 0, payload.data(), alpha.data()), "HQQ encoding");
    require(payload == expected_payload && alpha == expected_alpha, "HQQ storage bytes");
}

int main() {
    try {
        check_float_chunks();
        check_nonfinite();
        check_hqq_storage();
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
