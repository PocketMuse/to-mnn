#pragma once

#include "sd15/converter.hpp"

#include <exception>

namespace sd15::detail {

inline bool fail(ConvertError& error, ErrorCode code, const std::string& message,
                 const std::string& path = {}, const std::string& tensor = {}) {
    error = {code, message, path, tensor};
    return false;
}

inline void capture_exception(ConvertError& error, ErrorCode code,
                              const std::exception& exception) noexcept {
    try {
        error = {code, exception.what(), {}, {}};
    }
    catch (...) {
        error.code = ErrorCode::OutOfMemory;
        error.message.clear();
        error.path.clear();
        error.tensor.clear();
    }
}

}  // namespace sd15::detail
