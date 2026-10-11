#include "diffusion.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool ok) {
    if (!ok) {
        throw std::runtime_error("Runtime math test failed");
    }
}

template <class Function> void rejects(Function function) {
    try {
        function();
    }
    catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Invalid input was accepted");
}

int main() {
    try {
        using namespace sd15;
        SchedulerConfig config;
        const auto times = detail::timesteps(config, 20);
        require(times.size() == 20 && times.front() == 951 && times.back() == 1);
        rejects([&] { detail::timesteps(config, 0); });
        rejects([&] { detail::timesteps(config, 1000); });
        config.train_steps = 2;
        config.beta_start = 0.1;
        config.beta_end = 0.2;
        const auto a = detail::alphas(config);
        require(std::abs(a[0] - 0.9f) < 1e-6f && std::abs(a[1] - 0.72f) < 1e-6f);
        std::vector<float> latent{1.0f, -2.0f};
        detail::ddim_step(latent, {0.0f, 0.0f}, {0.0f, 0.0f}, 7.5f, 0.25f, 1.0f);
        require(latent[0] == 2.0f && latent[1] == -4.0f);
        latent = {1.0f};
        detail::ddim_step(latent, {0.1f}, {0.3f}, 2.0f, 0.25f, 1.0f);
        require(std::abs(latent[0] - 1.1339746f) < 1e-6f);
        rejects([&] { detail::ddim_step(latent, {}, {0.0f}, 1, 0.25f, 1); });
        GenerateOptions options;
        options.model_dir = "model";
        options.tokens.resize(kSequenceLength, 49407);
        options.negative_tokens = options.tokens;
        options.initial_latent.resize(kLatentElements);
        detail::validate(options);
        options.tokens[0] = kVocabularySize;
        rejects([&] { detail::validate(options); });
        options.tokens[0] = 49406;
        options.initial_latent[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { detail::validate(options); });
        std::vector<float> decoded(3 * kImageSize * kImageSize);
        decoded[0] = -2;
        decoded[kImageSize * kImageSize] = 0;
        decoded[2 * kImageSize * kImageSize] = 2;
        const auto image = detail::pixels(decoded);
        require(image.rgb[0] == 0 && image.rgb[1] == 128 && image.rgb[2] == 255);
        std::cout << "Runtime math and validation passed\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
