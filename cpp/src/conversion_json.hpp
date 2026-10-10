#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace sd15::detail {

using Json = nlohmann::json;
constexpr int kMaxJsonDepth = 64;
constexpr std::size_t kMaxTensorRank = 16;

/// 중복 key와 과도한 중첩을 거부하며 JSON을 읽는다.
inline Json parse_json(const std::vector<uint8_t>& bytes) {
    std::vector<std::unordered_set<std::string>> keys;
    return Json::parse(bytes.begin(), bytes.end(), [&keys](int depth, Json::parse_event_t event, Json& value) {
        if (depth > kMaxJsonDepth) {
            throw std::runtime_error("JSON nesting limit exceeded");
        }
        if (event == Json::parse_event_t::object_start) {
            keys.emplace_back();
        }
        else if (event == Json::parse_event_t::key) {
            if (!keys.back().insert(value.get<std::string>()).second) {
                throw std::runtime_error("Duplicate JSON key: " + value.get<std::string>());
            }
        }
        else if (event == Json::parse_event_t::object_end) {
            keys.pop_back();
        }
        return true;
    });
}

inline uint64_t unsigned_value(const Json& value) {
    if (!value.is_number_unsigned()) {
        throw std::runtime_error("Expected unsigned integer");
    }
    return value.get<uint64_t>();
}

inline uint64_t element_count(const Json& shape) {
    if (!shape.is_array() || shape.size() > kMaxTensorRank) {
        throw std::runtime_error("Invalid tensor shape");
    }
    uint64_t count = 1;
    for (const auto& dim : shape) {
        const uint64_t size = unsigned_value(dim);
        if (size != 0 && count > std::numeric_limits<uint64_t>::max() / size) {
            throw std::runtime_error("Tensor shape overflow");
        }
        count *= size;
    }
    return count;
}

inline std::size_t dtype_width(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") {
        return 4;
    }
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") {
        return 2;
    }
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") {
        return 8;
    }
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") {
        return 1;
    }
    throw std::runtime_error("Unsupported dtype: " + dtype);
}

} // namespace sd15::detail
