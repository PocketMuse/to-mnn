#include "sd15/runtime.hpp"
#include "diffusion.hpp"
#include "ddim_scheduler.hpp"
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
    detail::validate_model_options(options.preview);
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

Image Sd15Runtime::generate(const GenerateOptions& request, const InferenceProgress& progress,
                            const InferencePreview& preview) {
    return generate_impl(request, progress, nullptr, preview);
}

Image Sd15Runtime::generate(const GenerateOptions& request, GenerationCancellationToken& cancellation,
                            const InferenceProgress& progress, const InferencePreview& preview) {
    return generate_impl(request, progress, &cancellation, preview);
}

Image Sd15Runtime::generate_impl(const GenerateOptions& request, const InferenceProgress& progress,
                                 const GenerationCancellationToken* cancellation, const InferencePreview& preview) {
    if (impl_->busy.test_and_set(std::memory_order_acquire)) {
        throw std::runtime_error("Runtime is busy");
    }
    struct ResetBusy {
        std::atomic_flag& flag;
        ~ResetBusy() { flag.clear(std::memory_order_release); }
    } reset{impl_->busy};
    detail::check_cancellation(cancellation);
    // 콜백에서 호출자가 원래 요청을 바꾸어도 이번 생성의 설정은 유지한다.
    const auto options = request;
    detail::validate(options);
    if (options.preview_interval > 0 &&
        (!preview || !std::filesystem::is_regular_file(impl_->config.preview_decoder_path))) {
        throw std::invalid_argument("Previews require a TAESD model path and callback");
    }
    const auto seed = detail::resolve_seed(options.seed);
    const auto& root = impl_->root;
    const auto tokens = impl_->tokenizer.encode(options.prompt);
    const auto negative_tokens = impl_->tokenizer.encode(options.negative_prompt);
    const auto notify = [&](const char* stage, int step = 0) {
        detail::check_cancellation(cancellation);
        if (progress) {
            progress(stage, step, options.steps);
        }
        detail::check_cancellation(cancellation);
    };
    const detail::DdimScheduler scheduler(options.steps);
    std::vector<float> negative_context;
    std::vector<float> context;
    {
        notify("text_encoder");
        detail::ClipEncoder encoder(root / "text_encoder.mnn", impl_->config.clip, cancellation);
        negative_context = encoder.encode(negative_tokens);
        context = encoder.encode(tokens);
    }
    auto latent = detail::initial_noise(seed);
    for (auto& value : latent) {
        value *= scheduler.init_noise_sigma();
    }
    {
        notify("unet");
        detail::UNetDenoiser unet(root / "unet.mnn", impl_->config.unet, cancellation);
        std::unique_ptr<detail::PreviewDecoder> preview_decoder;
        std::vector<float> clean_prediction;
        for (std::size_t i = 0; i < scheduler.timesteps().size(); ++i) {
            const int time = scheduler.timesteps()[i];
            const auto& model_input = scheduler.scale_model_input(latent, i);
            auto noise = unet.predict(model_input, time, negative_context);
            const auto positive = unet.predict(model_input, time, context);
            detail::apply_guidance(noise, positive, options.guidance);
            const bool preview_step = options.preview_interval > 0 && (i + 1) % options.preview_interval == 0;
            scheduler.step(latent, noise, i, preview_step ? &clean_prediction : nullptr);
            const int completed = static_cast<int>(i + 1);
            notify("denoising", completed);
            if (preview_step) {
                if (!preview_decoder) {
                    preview_decoder = std::make_unique<detail::PreviewDecoder>(impl_->config.preview_decoder_path,
                                                                               impl_->config.preview, cancellation);
                }
                auto image = preview_decoder->decode(clean_prediction);
                image.seed = seed;
                detail::check_cancellation(cancellation);
                preview(completed, options.steps, image);
                detail::check_cancellation(cancellation);
            }
        }
    }
    notify("vae_decoder");
    detail::VaeDecoder decoder(root / "vae_decoder.mnn", impl_->config.vae, cancellation);
    auto image = decoder.decode(std::move(latent));
    image.seed = seed;
    detail::check_cancellation(cancellation);
    return image;
}

} // namespace sd15
