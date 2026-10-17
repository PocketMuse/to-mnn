#include "sd15_models.hpp"
#include "diffusion.hpp"

#include <cmath>

namespace sd15::detail {
namespace {

constexpr int kHiddenSize = 768;
constexpr float kVaeScalingFactor = 0.18215f;
const std::vector<int> kLatentShape{1, 4, 64, 64};
const std::vector<int> kContextShape{1, kSequenceLength, kHiddenSize};
const std::vector<int> kImageShape{1, 3, kImageSize, kImageSize};

} // namespace

ClipEncoder::ClipEncoder(const std::filesystem::path& path, const ModelOptions& options,
                         const GenerationCancellationToken* cancellation)
    : session_(path, 1, options, cancellation),
      tokens_(session_.bind_input<int32_t>("input_ids", {1, kSequenceLength})),
      context_(session_.bind_output("last_hidden_state", kContextShape)) {
}

std::vector<float> ClipEncoder::encode(const std::vector<int32_t>& tokens) {
    session_.input(tokens_, tokens);
    return session_.run(context_);
}

UNetDenoiser::UNetDenoiser(const std::filesystem::path& path, const ModelOptions& options,
                           const GenerationCancellationToken* cancellation)
    : session_(path, 3, options, cancellation), latent_(session_.bind_input<float>("sample", kLatentShape)),
      timestep_(session_.bind_input<int32_t>("timestep", {1})),
      context_(session_.bind_input<float>("encoder_hidden_states", kContextShape)),
      noise_(session_.bind_output("out_sample", kLatentShape)) {
}

std::vector<float> UNetDenoiser::predict(const std::vector<float>& latent, int timestep,
                                         const std::vector<float>& context) {
    // 실행 중 입력이 재사용될 수 있으므로 매 호출에 모두 복사한다.
    session_.input(latent_, latent);
    session_.input(timestep_, {timestep});
    session_.input(context_, context);
    return session_.run(noise_);
}

VaeDecoder::VaeDecoder(const std::filesystem::path& path, const ModelOptions& options,
                       const GenerationCancellationToken* cancellation)
    : session_(path, 1, options, cancellation), latent_(session_.bind_input<float>("latent_sample", kLatentShape)),
      image_(session_.bind_output("sample", kImageShape)) {
}

PreviewDecoder::PreviewDecoder(const std::filesystem::path& path, const ModelOptions& options,
                               const GenerationCancellationToken* cancellation)
    : session_(path, 1, options, cancellation), latent_(session_.bind_input<float>("latent_sample", kLatentShape)),
      image_(session_.bind_output("sample", kImageShape)) {
}

Image PreviewDecoder::decode(const std::vector<float>& latent) {
    // 원본 TAESD는 diffusion latent를 직접 받고 [0, 1] RGB를 반환한다.
    session_.input(latent_, latent);
    auto decoded = session_.run(image_);
    for (auto& value : decoded) {
        value = value * 2.0f - 1.0f;
    }
    return pixels(decoded);
}

Image VaeDecoder::decode(std::vector<float> latent) {
    for (auto& value : latent) {
        value /= kVaeScalingFactor;
        if (!std::isfinite(value)) {
            throw std::runtime_error("Non-finite VAE input");
        }
    }
    session_.input(latent_, latent);
    return pixels(session_.run(image_));
}

} // namespace sd15::detail
