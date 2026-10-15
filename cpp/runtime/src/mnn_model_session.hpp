#pragma once

#include "sd15/runtime.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace sd15::detail {

struct ModelOptions {
    Precision precision = Precision::High;
    Tuning tuning = Tuning::None;
    bool low_memory = false;
};

void validate_model_options(const ModelOptions& options);
void validate_mnn_version();

// 모델 종류를 알지 않는다. Interpreter가 Session까지 소유한다.
class MnnModelSession {
public:
    MnnModelSession(const std::filesystem::path& path, std::size_t input_count, const ModelOptions& options);

    template <typename T> void input(const char* name, const std::vector<int>& shape, const std::vector<T>& values) {
        auto* tensor = interpreter_->getSessionInput(session_, name);
        check<T>(tensor, name, shape);
        if (tensor->elementSize() != static_cast<int>(values.size())) {
            throw std::runtime_error(std::string("Input size mismatch: ") + name);
        }
        std::unique_ptr<MNN::Tensor> host(MNN::Tensor::create<T>(shape, nullptr, MNN::Tensor::CAFFE));
        if (!host || host->host<T>() == nullptr) {
            throw std::runtime_error("Cannot allocate host input");
        }
        std::copy(values.begin(), values.end(), host->host<T>());
        if (!tensor->copyFromHostTensor(host.get())) {
            throw std::runtime_error(std::string("Cannot copy input: ") + name);
        }
    }

    std::vector<float> run(const char* name, const std::vector<int>& shape);

private:
    template <typename T> static void check(MNN::Tensor* tensor, const char* name, const std::vector<int>& shape) {
        if (tensor == nullptr || tensor->shape() != shape) {
            throw std::runtime_error(std::string("Missing tensor or shape mismatch: ") + name);
        }
        const auto actual = tensor->getType();
        const auto expected = halide_type_of<T>();
        if (actual.code != expected.code || actual.bits != expected.bits || actual.lanes != expected.lanes) {
            throw std::runtime_error(std::string("Tensor dtype mismatch: ") + name);
        }
    }

    std::unique_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
};

} // namespace sd15::detail
