#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd15::detail {

// 고정 SD1.5 CLIP 어휘를 사용한다. 생성 간 prompt 캐시는 보관하지 않는다.
class ClipTokenizer {
public:
    explicit ClipTokenizer(const std::string& directory);
    std::vector<int32_t> encode(const std::string& prompt) const;

private:
    std::unordered_map<std::string, int32_t> vocabulary_;
    std::unordered_map<std::string, int> ranks_;
    std::array<std::string, 256> bytes_;
};

} // namespace sd15::detail
