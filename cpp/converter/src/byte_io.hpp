#pragma once

#include <cstddef>
#include <cstdint>

namespace sd15::detail {

inline constexpr uint32_t kBitsPerByte = 8;

// 호출자는 size바이트를 제공한다. size는 최대 sizeof(uint64_t)다.
inline uint64_t read_le(const uint8_t* bytes, std::size_t size) {
    uint64_t value = 0;
    for (std::size_t i = 0; i < size; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (i * kBitsPerByte);
    }
    return value;
}

} // namespace sd15::detail
