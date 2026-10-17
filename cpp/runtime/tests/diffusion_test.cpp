#include "diffusion.hpp"

#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>

using namespace sd15;

TEST_CASE("DDIM timesteps and alphas match the schedule") {
    const auto times = detail::timesteps(20);
    REQUIRE(times.size() == 20);
    CHECK(times.front() == 951);
    CHECK(times.back() == 1);
    REQUIRE_THROWS_AS(detail::timesteps(0), std::invalid_argument);
    REQUIRE_THROWS_AS(detail::timesteps(1000), std::invalid_argument);
    const auto a = detail::alphas();
    REQUIRE(a.size() == 1000);
    CHECK(std::abs(a.front() - 0.99915f) < 1e-6f);
    // SD1.5 scaled-linear schedule의 독립적인 double 기준값과 비교한다.
    double expected = 1;
    for (int i = 0; i < 1000; ++i) {
        const double fraction = i / 999.0;
        const double root = (1 - fraction) * std::sqrt(0.00085) + fraction * std::sqrt(0.012);
        expected *= 1 - root * root;
        CHECK(std::abs(a[i] - expected) < 2e-6);
    }
}

TEST_CASE("DDIM schedules keep every supported step count within the alpha table") {
    for (int steps = 1; steps < 1000; ++steps) {
        GenerateOptions options;
        options.steps = steps;
        REQUIRE_NOTHROW(detail::validate(options));
        const auto times = detail::timesteps(steps);
        REQUIRE(times.size() == static_cast<std::size_t>(steps));
        REQUIRE(times.back() == 1);
        REQUIRE(times.front() < 1000);
        for (std::size_t i = 1; i < times.size(); ++i) {
            REQUIRE(times[i - 1] - times[i] == 1000 / steps);
        }
    }
}

TEST_CASE("DDIM applies guidance") {
    std::vector<float> latent{1.0f, -2.0f};
    detail::ddim_step(latent, {0.0f, 0.0f}, {0.0f, 0.0f}, 7.5f, 0.25f, 1.0f);
    CHECK(latent[0] == 2.0f);
    CHECK(latent[1] == -4.0f);
    latent = {1.0f};
    detail::ddim_step(latent, {0.1f}, {0.3f}, 2.0f, 0.25f, 1.0f);
    CHECK(std::abs(latent[0] - 1.1339746f) < 1e-6f);
}

TEST_CASE("Denoising preserves latent and preview buffer sizes across steps") {
    auto latent = detail::initial_noise(42);
    const std::vector<float> negative(kLatentElements, 0.1f);
    const std::vector<float> positive(kLatentElements, 0.3f);
    std::vector<float> clean;
    const auto alpha = detail::alphas();
    const auto times = detail::timesteps(20);
    for (std::size_t i = 0; i < times.size(); ++i) {
        const int previous = times[i] - detail::kTrainSteps / 20;
        const bool preview = i % 2 == 0;
        detail::ddim_step(latent, negative, positive, 7.5f, alpha[times[i]],
                          previous >= 0 ? alpha[previous] : alpha.front(), preview ? &clean : nullptr);
        REQUIRE(latent.size() == kLatentElements);
        REQUIRE(clean.size() == kLatentElements);
    }
}

TEST_CASE("Seeds preserve explicit values and resolve negative values") {
    GenerateOptions options;
    CHECK(options.seed == -1);
    CHECK(detail::resolve_seed(0) == 0);
    CHECK(detail::resolve_seed(42) == 42);
    CHECK(detail::resolve_seed(4294967295LL) == std::numeric_limits<uint32_t>::max());
    REQUIRE_THROWS_AS(detail::resolve_seed(4294967296LL), std::invalid_argument);
    // 음수 경로의 실제 seed로 같은 노이즈를 다시 생성할 수 있다.
    for (const int64_t value : std::vector<int64_t>{-1, -42, std::numeric_limits<int64_t>::min()}) {
        const auto seed = detail::resolve_seed(value);
        CHECK(detail::initial_noise(seed) == detail::initial_noise(detail::resolve_seed(seed)));
    }
    const auto first_seed = detail::resolve_seed(-1);
    bool varied = false;
    for (int i = 0; i < 8; ++i) {
        varied |= detail::resolve_seed(-1) != first_seed;
    }
    CHECK(varied);
}

TEST_CASE("DDIM exposes the clean prediction for previews") {
    std::vector<float> latent{1.0f};
    std::vector<float> clean;
    detail::ddim_step(latent, {0.1f}, {0.3f}, 2.0f, 0.25f, 0.81f, &clean);
    REQUIRE(clean.size() == 1);
    CHECK(std::abs(clean[0] - 1.1339746f) < 1e-6f);
    CHECK(std::abs(latent[0] - (0.9f * clean[0] + std::sqrt(0.19f) * 0.5f)) < 1e-6f);
}

TEST_CASE("Generation options reject invalid seeds steps and guidance") {
    GenerateOptions options;
    options.seed = 4294967296LL;
    REQUIRE_THROWS_AS(detail::validate(options), std::invalid_argument);
    options.seed = -1;
    options.prompt = "a cat";
    detail::validate(options);
    options.steps = 0;
    REQUIRE_THROWS_AS(detail::validate(options), std::invalid_argument);
    options.steps = 1000;
    REQUIRE_THROWS_AS(detail::validate(options), std::invalid_argument);
    options.steps = 20;
    options.guidance = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(detail::validate(options), std::invalid_argument);
}

TEST_CASE("Initial noise is reproducible and approximately normal") {
    const auto noise = detail::initial_noise(42);
    CHECK(noise.size() == kLatentElements);
    CHECK(noise == detail::initial_noise(42));
    CHECK(noise != detail::initial_noise(123));
    double mean = 0, square = 0;
    for (const auto value : noise) {
        CHECK(std::isfinite(value));
        mean += value;
        square += value * value;
    }
    CHECK(std::abs(mean / noise.size()) < 0.05);
    CHECK(std::abs(square / noise.size() - 1.0) < 0.1);
}

TEST_CASE("Pixels clamp and round decoded channels to RGB8") {
    std::vector<float> decoded(3 * kImageSize * kImageSize);
    decoded[0] = -2;
    decoded[kImageSize * kImageSize] = 0;
    decoded[2 * kImageSize * kImageSize] = 2;
    // RGB8 값 128.25, 128.5, 128.75: .5는 위 정수로 반올림한다.
    decoded[1] = 1.5f / 255;
    decoded[2] = 2.0f / 255;
    decoded[3] = 2.5f / 255;
    const auto image = detail::pixels(decoded);
    REQUIRE(image.rgb.size() == decoded.size());
    CHECK(image.rgb[0] == 0);
    CHECK(image.rgb[1] == 128);
    CHECK(image.rgb[2] == 255);
    CHECK(image.rgb[3] == 128);
    CHECK(image.rgb[6] == 129);
    CHECK(image.rgb[9] == 129);
}

TEST_CASE("Runtime math rejects nonfinite results with valid buffer sizes") {
    std::vector<float> latent(kLatentElements, 1.0f);
    std::vector<float> noise(kLatentElements, 0.0f);
    noise.back() = std::numeric_limits<float>::infinity();
    REQUIRE_THROWS_AS(detail::ddim_step(latent, noise, noise, 7.5f, 0.25f, 1.0f), std::runtime_error);

    std::vector<float> decoded(3 * kImageSize * kImageSize, 0.0f);
    decoded.back() = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(detail::pixels(decoded), std::runtime_error);
}
