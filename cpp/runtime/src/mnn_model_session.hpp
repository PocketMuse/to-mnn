#pragma once

#include "sd15/runtime.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace sd15::detail {

void validate_model_options(const ModelOptions& options);
void validate_mnn_version();
inline void check_cancellation(const GenerationCancellationToken* cancellation) {
    if (cancellation != nullptr && cancellation->is_requested()) {
        throw GenerationCancelled{};
    }
}

// 모델 종류를 알지 않는다. Interpreter가 Session까지 소유한다.
// ModelOptions는 Sd15Runtime 생성 시 검증한다.
class MnnModelSession {
public:
    // 연결은 세션과 같은 수명이며 세션 resize 없이 재사용한다.
    template <typename T> struct Input {
        MNN::Tensor* tensor;
        std::string name;
        std::vector<int> shape;
    };

    struct Output {
        MNN::Tensor* tensor;
        std::string name;
    };

    MnnModelSession(const std::filesystem::path& path, std::size_t input_count, const ModelOptions& options,
                    const GenerationCancellationToken* cancellation = nullptr);

    template <typename T> Input<T> bind_input(const char* name, const std::vector<int>& shape) {
        check_cancellation(cancellation_);
        auto* tensor = interpreter_->getSessionInput(session_, name);
        check<T>(tensor, name, shape);
        return {tensor, name, shape};
    }

    Output bind_output(const char* name, const std::vector<int>& shape);

    template <typename T> void input(const Input<T>& binding, const std::vector<T>& values) {
        check_cancellation(cancellation_);
        if (binding.tensor->elementSize() != static_cast<int>(values.size())) {
            throw std::runtime_error("Input size mismatch: " + binding.name);
        }
        std::unique_ptr<MNN::Tensor> host(MNN::Tensor::create<T>(binding.shape, nullptr, MNN::Tensor::CAFFE));
        if (!host || host->host<T>() == nullptr) {
            throw std::runtime_error("Cannot allocate host input");
        }
        std::copy(values.begin(), values.end(), host->host<T>());
        if (!binding.tensor->copyFromHostTensor(host.get())) {
            throw std::runtime_error("Cannot copy input: " + binding.name);
        }
    }

    std::vector<float> run(const Output& binding);

private:
    const GenerationCancellationToken* cancellation_;
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
