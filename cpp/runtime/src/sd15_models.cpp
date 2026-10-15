#include "sd15_models.hpp"
#include "diffusion.hpp"

#include <cmath>

namespace sd15::detail {
namespace {

constexpr int kHiddenSize = 768;
constexpr float kVaeScalingFactor = 0.18215f;
const std::vector<int> kLatentShape{1, 4, 64, 64};
const std::vector<int> kContextShape{1, kSequenceLength, kHiddenSize};

} // namespace

ClipEncoder::ClipEncoder(const std::filesystem::path& path, const ModelOptions& options,
                         const GenerationCancellationToken* cancellation)
    : session_(path, 1, options, cancellation) {
}

std::vector<float> ClipEncoder::encode(const std::vector<int32_t>& tokens) {
    session_.input("input_ids", {1, kSequenceLength}, tokens);
    return session_.run("last_hidden_state", kContextShape);
}

UNetDenoiser::UNetDenoiser(const std::filesystem::path& path, const ModelOptions& options,
                           const GenerationCancellationToken* cancellation)
    : session_(path, 3, options, cancellation) {
}

std::vector<float> UNetDenoiser::predict(const std::vector<float>& latent, int timestep,
                                         const std::vector<float>& context) {
    // 실행 중 입력이 재사용될 수 있으므로 매 호출에 모두 복사한다.
    session_.input("sample", kLatentShape, latent);
    session_.input<int32_t>("timestep", {1}, {timestep});
    session_.input("encoder_hidden_states", kContextShape, context);
    return session_.run("out_sample", kLatentShape);
}

VaeDecoder::VaeDecoder(const std::filesystem::path& path, const ModelOptions& options,
                       const GenerationCancellationToken* cancellation)
    : session_(path, 1, options, cancellation) {
}

Image VaeDecoder::decode(std::vector<float> latent) {
    for (auto& value : latent) {
        value /= kVaeScalingFactor;
        if (!std::isfinite(value)) {
            throw std::runtime_error("Non-finite VAE input");
        }
    }
    session_.input("latent_sample", kLatentShape, latent);
    return pixels(session_.run("sample", {1, 3, kImageSize, kImageSize}));
}

} // namespace sd15::detail
