#include "mnn_model_session.hpp"

#include <array>
#include <cmath>

namespace sd15::detail {
namespace {

MNN::BackendConfig::PrecisionMode precision_mode(Precision value) {
    switch (value) {
    case Precision::High:
        return MNN::BackendConfig::Precision_High;
    case Precision::Normal:
        return MNN::BackendConfig::Precision_Normal;
    case Precision::Low:
        return MNN::BackendConfig::Precision_Low;
    }
    throw std::invalid_argument("Invalid model precision");
}

int tuning_mode(Tuning value) {
    switch (value) {
    case Tuning::None:
        return MNN_GPU_TUNING_NONE;
    case Tuning::Fast:
        return MNN_GPU_TUNING_FAST;
    case Tuning::Normal:
        return MNN_GPU_TUNING_NORMAL;
    case Tuning::Wide:
        return MNN_GPU_TUNING_WIDE;
    }
    throw std::invalid_argument("Invalid tuning mode");
}

} // namespace

void validate_model_options(const ModelOptions& options) {
    precision_mode(options.precision);
    tuning_mode(options.tuning);
#ifndef SD15_MNN_LOW_MEMORY
    if (options.low_memory) {
        throw std::invalid_argument("Low-memory requires an MNN_LOW_MEMORY build");
    }
#endif
}

void validate_mnn_version() {
    if (std::string(MNN::getVersion()) != "3.6.1") {
        throw std::runtime_error("MNN 3.6.1 is required");
    }
}

MnnModelSession::MnnModelSession(const std::filesystem::path& path, std::size_t input_count,
                                 const ModelOptions& options) {
    validate_model_options(options);
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("Missing model: " + path.string());
    }
    interpreter_.reset(MNN::Interpreter::createFromFile(path.string().c_str()));
    if (!interpreter_) {
        throw std::runtime_error("Cannot load model: " + path.string());
    }
    const auto external = path.string() + ".weight";
    if (std::filesystem::is_regular_file(external)) {
        interpreter_->setExternalFile(external.c_str());
    }
    MNN::BackendConfig backend;
    backend.precision = precision_mode(options.precision);
    backend.memory = options.low_memory ? MNN::BackendConfig::Memory_Low : MNN::BackendConfig::Memory_Normal;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;
    config.mode = tuning_mode(options.tuning) | MNN_GPU_MEMORY_BUFFER;
    config.backendConfig = &backend;
    session_ = interpreter_->createSession(config);
    if (session_ == nullptr) {
        throw std::runtime_error("Cannot create OpenCL session: " + path.string());
    }
    std::array<int, 2> backends{};
    if (!interpreter_->getSessionInfo(session_, MNN::Interpreter::BACKENDS, backends.data()) ||
        backends[0] != MNN_FORWARD_OPENCL) {
        throw std::runtime_error("OpenCL unavailable; CPU fallback rejected: " + path.string());
    }
    if (interpreter_->getSessionInputAll(session_).size() != input_count) {
        throw std::runtime_error("Unexpected model input count: " + path.string());
    }
}

std::vector<float> MnnModelSession::run(const char* name, const std::vector<int>& shape) {
    if (interpreter_->runSession(session_) != MNN::NO_ERROR) {
        throw std::runtime_error("MNN inference failed");
    }
    auto* tensor = interpreter_->getSessionOutput(session_, name);
    check<float>(tensor, name, shape);
    MNN::Tensor host(tensor, MNN::Tensor::CAFFE);
    if (host.host<float>() == nullptr || !tensor->copyToHostTensor(&host)) {
        throw std::runtime_error("Cannot copy output");
    }
    const float* data = host.host<float>();
    std::vector<float> result(data, data + host.elementSize());
    if (std::any_of(result.begin(), result.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error(std::string("Non-finite output: ") + name);
    }
    return result;
}

} // namespace sd15::detail
