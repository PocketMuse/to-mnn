#include "diffusion.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sd15::detail {
namespace {

void check_scheduler(const SchedulerConfig& config) {
    if (config.train_steps < 2 || config.train_steps > 10000 || config.steps_offset < 0 ||
        config.steps_offset >= config.train_steps || !std::isfinite(config.beta_start) ||
        !std::isfinite(config.beta_end) || config.beta_start <= 0 || config.beta_start > config.beta_end ||
        config.beta_end >= 1) {
        throw std::invalid_argument("Invalid scaled-linear scheduler configuration");
    }
}

} // namespace

std::vector<int> timesteps(const SchedulerConfig& config, int steps) {
    check_scheduler(config);
    if (steps < 1 || steps >= config.train_steps) {
        throw std::invalid_argument("steps must be in [1, train_steps)");
    }
    const int stride = config.train_steps / steps;
    std::vector<int> values;
    for (int i = steps - 1; i >= 0; --i) {
        const int time = i * stride + config.steps_offset;
        if (time >= config.train_steps) {
            throw std::invalid_argument("Timestep outside training schedule");
        }
        values.push_back(time);
    }
    return values;
}

void validate(const GenerateOptions& options) {
    timesteps(options.scheduler, options.steps);
    if (options.model_dir.empty() || !std::isfinite(options.guidance) || !std::isfinite(options.vae_scaling_factor) ||
        options.vae_scaling_factor <= 0) {
        throw std::invalid_argument("Invalid model path, guidance or VAE scaling factor");
    }
    for (const auto* tokens : {&options.tokens, &options.negative_tokens}) {
        if (tokens->size() != kSequenceLength ||
            std::any_of(tokens->begin(), tokens->end(), [](int32_t id) { return id < 0 || id >= kVocabularySize; })) {
            throw std::invalid_argument("Expected 77 CLIP token IDs in [0, 49408)");
        }
    }
    if (options.initial_latent.size() != kLatentElements ||
        std::any_of(options.initial_latent.begin(), options.initial_latent.end(),
                    [](float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("Expected finite initial latent with shape [1, 4, 64, 64]");
    }
}

std::vector<float> alphas(const SchedulerConfig& config) {
    check_scheduler(config);
    const double first = std::sqrt(config.beta_start);
    const double step = (std::sqrt(config.beta_end) - first) / (config.train_steps - 1);
    float product = 1;
    std::vector<float> result;
    for (int i = 0; i < config.train_steps; ++i) {
        const float root = static_cast<float>(first + i * step);
        product *= 1.0f - root * root;
        if (!std::isfinite(product) || product <= 0) {
            throw std::invalid_argument("Invalid cumulative alpha");
        }
        result.push_back(product);
    }
    return result;
}

void ddim_step(std::vector<float>& latent, const std::vector<float>& unconditional,
               const std::vector<float>& conditional, float guidance, float alpha, float previous_alpha) {
    if (latent.size() != unconditional.size() || latent.size() != conditional.size() ||
        !(alpha > 0 && alpha <= 1 && previous_alpha > 0 && previous_alpha <= 1) || !std::isfinite(guidance)) {
        throw std::invalid_argument("Invalid DDIM inputs");
    }
    const float signal = std::sqrt(alpha);
    const float noise_scale = std::sqrt(1 - alpha);
    const float previous_signal = std::sqrt(previous_alpha);
    const float previous_noise = std::sqrt(1 - previous_alpha);
    for (std::size_t i = 0; i < latent.size(); ++i) {
        const float noise = unconditional[i] + guidance * (conditional[i] - unconditional[i]);
        const float clean = (latent[i] - noise_scale * noise) / signal;
        latent[i] = previous_signal * clean + previous_noise * noise;
        if (!std::isfinite(latent[i])) {
            throw std::runtime_error("Non-finite DDIM latent");
        }
    }
}

Image pixels(const std::vector<float>& decoded) {
    constexpr std::size_t plane = kImageSize * kImageSize;
    if (decoded.size() != 3 * plane) {
        throw std::invalid_argument("Expected decoded shape [1, 3, 512, 512]");
    }
    Image image;
    image.rgb.resize(decoded.size());
    for (std::size_t i = 0; i < plane; ++i) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const float value = decoded[channel * plane + i];
            if (!std::isfinite(value)) {
                throw std::runtime_error("Non-finite decoded image");
            }
            const float scaled = std::clamp(value / 2 + 0.5f, 0.0f, 1.0f) * 255;
            // NumPy rint와 같은 ties-to-even 반올림.
            const int lower = static_cast<int>(scaled);
            const float fraction = scaled - lower;
            image.rgb[3 * i + channel] =
                static_cast<uint8_t>(lower + (fraction > 0.5f || (fraction == 0.5f && lower % 2 != 0) ? 1 : 0));
        }
    }
    return image;
}

} // namespace sd15::detail
