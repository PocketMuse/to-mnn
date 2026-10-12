#include "sd15/runtime.hpp"
#include "diffusion.hpp"
#include "clip_tokenizer.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace sd15 {
namespace {

constexpr int kHiddenSize = 768;
constexpr float kVaeScalingFactor = 0.18215f;
const std::vector<int> kLatentShape{1, 4, 64, 64};
const std::vector<int> kContextShape{1, kSequenceLength, kHiddenSize};

// Interpreter가 Session까지 소유한다. 단계 종료 시 엔진에 자원을 반환한다.
class Model {
public:
    Model(const std::filesystem::path& path, std::size_t input_count) {
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
        backend.precision = MNN::BackendConfig::Precision_High;
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_OPENCL;
        config.mode = MNN_GPU_TUNING_NONE | MNN_GPU_MEMORY_BUFFER;
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

    std::vector<float> run(const char* name, const std::vector<int>& shape) {
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

} // namespace

struct Sd15Runtime::Impl {
    explicit Impl(const RuntimeOptions& options)
        : root(options.model_dir),
          tokenizer(options.tokenizer_dir.empty() ? (root / "tokenizer").string() : options.tokenizer_dir) {}

    const std::filesystem::path root;
    const detail::ClipTokenizer tokenizer;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
};

Sd15Runtime::Sd15Runtime(const RuntimeOptions& options) {
    if (options.model_dir.empty()) {
        throw std::invalid_argument("Model directory is required");
    }
    if (std::string(MNN::getVersion()) != "3.6.1") {
        throw std::runtime_error("MNN 3.6.1 is required");
    }
    const std::filesystem::path root(options.model_dir);
    for (const char* name : {"text_encoder.mnn", "unet.mnn", "vae_decoder.mnn"}) {
        if (!std::filesystem::is_regular_file(root / name)) {
            throw std::runtime_error("Missing model: " + (root / name).string());
        }
    }
    impl_ = std::make_unique<Impl>(options);
}

Sd15Runtime::~Sd15Runtime() = default;

Image Sd15Runtime::generate(const GenerateOptions& request, const InferenceProgress& progress) {
    if (impl_->busy.test_and_set(std::memory_order_acquire)) {
        throw std::runtime_error("Runtime is busy");
    }
    struct ResetBusy {
        std::atomic_flag& flag;
        ~ResetBusy() { flag.clear(std::memory_order_release); }
    } reset{impl_->busy};
    // 콜백에서 호출자가 원래 요청을 바꾸어도 이번 생성의 설정은 유지한다.
    const auto options = request;
    detail::validate(options);
    const auto& root = impl_->root;
    const auto tokens = impl_->tokenizer.encode(options.prompt);
    const auto negative_tokens = impl_->tokenizer.encode(options.negative_prompt);
    const detail::SchedulerConfig scheduler;
    const auto notify = [&](const char* stage, int step = 0) {
        if (progress) {
            progress(stage, step, options.steps);
        }
    };
    const auto alpha = detail::alphas(scheduler);
    const auto times = detail::timesteps(scheduler, options.steps);
    std::vector<float> negative_context;
    std::vector<float> context;
    {
        notify("text_encoder");
        Model encoder(root / "text_encoder.mnn", 1);
        encoder.input("input_ids", {1, kSequenceLength}, negative_tokens);
        negative_context = encoder.run("last_hidden_state", kContextShape);
        encoder.input("input_ids", {1, kSequenceLength}, tokens);
        context = encoder.run("last_hidden_state", kContextShape);
    }
    auto latent = detail::initial_noise(options.seed);
    {
        notify("unet");
        Model unet(root / "unet.mnn", 3);
        for (int i = 0; i < options.steps; ++i) {
            const int time = times[i];
            unet.input("sample", kLatentShape, latent);
            unet.input<int32_t>("timestep", {1}, {time});
            unet.input("encoder_hidden_states", kContextShape, negative_context);
            const auto negative = unet.run("out_sample", kLatentShape);
            // 실행 중 입력 버퍼가 재사용될 수 있으므로 매 호출에 다시 채운다.
            unet.input("sample", kLatentShape, latent);
            unet.input<int32_t>("timestep", {1}, {time});
            unet.input("encoder_hidden_states", kContextShape, context);
            const auto positive = unet.run("out_sample", kLatentShape);
            const int previous = time - scheduler.train_steps / options.steps;
            const float previous_alpha =
                previous >= 0 ? alpha[previous] : (scheduler.set_alpha_to_one ? 1.0f : alpha.front());
            detail::ddim_step(latent, negative, positive, options.guidance, alpha[time], previous_alpha);
            notify("denoising", i + 1);
        }
    }
    notify("vae_decoder");
    for (auto& value : latent) {
        value /= kVaeScalingFactor;
        if (!std::isfinite(value)) {
            throw std::runtime_error("Non-finite VAE input");
        }
    }
    Model decoder(root / "vae_decoder.mnn", 1);
    decoder.input("latent_sample", kLatentShape, latent);
    return detail::pixels(decoder.run("sample", {1, 3, kImageSize, kImageSize}));
}

} // namespace sd15
