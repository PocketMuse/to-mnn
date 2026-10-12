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
        detail::SchedulerConfig config;
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
        require(options.seed == -1);
        require(detail::resolve_seed(0) == 0);
        require(detail::resolve_seed(42) == 42);
        require(detail::resolve_seed(4294967295LL) == std::numeric_limits<uint32_t>::max());
        rejects([&] { detail::resolve_seed(4294967296LL); });
        // 음수 경로의 실제 seed로 같은 노이즈를 다시 생성할 수 있다.
        for (const int64_t value : std::vector<int64_t>{-1, -42, std::numeric_limits<int64_t>::min()}) {
            const auto seed = detail::resolve_seed(value);
            require(detail::initial_noise(seed) == detail::initial_noise(detail::resolve_seed(seed)));
        }
        const auto first_seed = detail::resolve_seed(-1);
        bool varied = false;
        for (int i = 0; i < 8; ++i) {
            varied |= detail::resolve_seed(-1) != first_seed;
        }
        require(varied);
        options.seed = 4294967296LL;
        rejects([&] { detail::validate(options); });
        options.seed = -1;
        options.prompt = "a cat";
        detail::validate(options);
        options.steps = 0;
        rejects([&] { detail::validate(options); });
        options.steps = 20;
        options.guidance = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { detail::validate(options); });
        const auto noise = detail::initial_noise(42);
        require(noise.size() == kLatentElements);
        require(noise == detail::initial_noise(42));
        require(noise != detail::initial_noise(123));
        double mean = 0, square = 0;
        for (const auto value : noise) {
            require(std::isfinite(value));
            mean += value;
            square += value * value;
        }
        require(std::abs(mean / noise.size()) < 0.05);
        require(std::abs(square / noise.size() - 1.0) < 0.1);
        std::vector<float> decoded(3 * kImageSize * kImageSize);
        decoded[0] = -2;
        decoded[kImageSize * kImageSize] = 0;
        decoded[2 * kImageSize * kImageSize] = 2;
        // RGB8 값 128.25, 128.5, 128.75: .5는 위 정수로 반올림한다.
        decoded[1] = 1.5f / 255;
        decoded[2] = 2.0f / 255;
        decoded[3] = 2.5f / 255;
        const auto image = detail::pixels(decoded);
        require(image.rgb[0] == 0 && image.rgb[1] == 128 && image.rgb[2] == 255);
        require(image.rgb[3] == 128 && image.rgb[6] == 129 && image.rgb[9] == 129);
        std::cout << "Runtime math and validation passed\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
