#include "sd15/runtime.hpp"
#include "diffusion.hpp"
#include "clip_tokenizer.hpp"
#include "sd15_models.hpp"

#include <atomic>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sd15 {

struct Sd15Runtime::Impl {
    explicit Impl(const RuntimeOptions& options)
        : config(options), root(options.model_dir),
          tokenizer(options.tokenizer_dir.empty() ? (root / "tokenizer").string() : options.tokenizer_dir) {}

    const RuntimeOptions config;
    const std::filesystem::path root;
    const detail::ClipTokenizer tokenizer;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
};

Sd15Runtime::Sd15Runtime(const RuntimeOptions& options) {
    detail::validate_model_options(options.clip);
    detail::validate_model_options(options.unet);
    detail::validate_model_options(options.vae);
    if (options.model_dir.empty()) {
        throw std::invalid_argument("Model directory is required");
    }
    detail::validate_mnn_version();
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
    const auto seed = detail::resolve_seed(options.seed);
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
        detail::ClipEncoder encoder(root / "text_encoder.mnn", impl_->config.clip);
        negative_context = encoder.encode(negative_tokens);
        context = encoder.encode(tokens);
    }
    auto latent = detail::initial_noise(seed);
    {
        notify("unet");
        detail::UNetDenoiser unet(root / "unet.mnn", impl_->config.unet);
        for (int i = 0; i < options.steps; ++i) {
            const int time = times[i];
            const auto negative = unet.predict(latent, time, negative_context);
            const auto positive = unet.predict(latent, time, context);
            const int previous = time - scheduler.train_steps / options.steps;
            const float previous_alpha =
                previous >= 0 ? alpha[previous] : (scheduler.set_alpha_to_one ? 1.0f : alpha.front());
            detail::ddim_step(latent, negative, positive, options.guidance, alpha[time], previous_alpha);
            notify("denoising", i + 1);
        }
    }
    notify("vae_decoder");
    detail::VaeDecoder decoder(root / "vae_decoder.mnn", impl_->config.vae);
    auto image = decoder.decode(std::move(latent));
    image.seed = seed;
    return image;
}

} // namespace sd15
