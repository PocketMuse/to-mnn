#include "diffusion.hpp"
#include "ddim_scheduler.hpp"

#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "legacy_ddim.hpp"

using namespace sd15;

static_assert(!std::is_default_constructible_v<GenerateOptions>);

TEST_CASE("Generation requires explicit sampling steps and guidance") {
    GenerateOptions options(SamplingPreset::Dpmpp2mKarras, 33, 4.5f);
    CHECK(options.sampling == SamplingPreset::Dpmpp2mKarras);
    CHECK(options.steps == 33);
    CHECK(options.guidance == 4.5f);
    CHECK(options.seed == -1);
    CHECK(options.preview_interval == 0);
    REQUIRE_NOTHROW(detail::validate(options));
}

TEST_CASE("DDIM scheduler preserves the legacy trajectory and previews") {
    for (const int steps : {1, 2, 20, 33, 999}) {
        detail::DdimScheduler scheduler(steps);
        const auto alpha = legacy::alphas();
        auto expected = detail::initial_noise(42);
        auto actual = expected;
        std::vector<float> expected_clean;
        std::vector<float> actual_clean;
        const std::vector<float> negative(kLatentElements, 0.1f);
        const std::vector<float> positive(kLatentElements, 0.3f);
        auto noise = negative;
        detail::apply_guidance(noise, positive, 7.5f);
        CHECK(scheduler.init_noise_sigma() == 1.0f);
        for (std::size_t i = 0; i < scheduler.timesteps().size(); ++i) {
            const int time = scheduler.timesteps()[i];
            const int previous = time - detail::kTrainSteps / steps;
            const bool preview = i % 2 == 0;
            legacy::ddim_step(expected, negative, positive, 7.5f, alpha[time],
                              previous >= 0 ? alpha[previous] : alpha.front(), preview ? &expected_clean : nullptr);
            CHECK(scheduler.scale_model_input(actual, i) == actual);
            scheduler.step(actual, noise, i, preview ? &actual_clean : nullptr);
            REQUIRE(actual == expected);
            REQUIRE(actual_clean == expected_clean);
        }
    }
}

TEST_CASE("DDIM timesteps and alphas match the schedule") {
    detail::DdimScheduler scheduler(20);
    const auto& times = scheduler.timesteps();
    REQUIRE(times.size() == 20);
    CHECK(times.front() == 951);
    CHECK(times.back() == 1);
    REQUIRE_THROWS_AS(detail::DdimScheduler(0), std::invalid_argument);
    REQUIRE_THROWS_AS(detail::DdimScheduler(1000), std::invalid_argument);
    // 독립적인 double 기준으로 모든 학습 시점의 alpha를 계산한다.
    std::vector<double> alpha;
    double product = 1;
    for (int i = 0; i < 1000; ++i) {
        const double fraction = i / 999.0;
        const double root = (1 - fraction) * std::sqrt(0.00085) + fraction * std::sqrt(0.012);
        product *= 1 - root * root;
        alpha.push_back(product);
    }
    detail::DdimScheduler dense(999);
    for (std::size_t i = 0; i < dense.timesteps().size(); ++i) {
        const int time = dense.timesteps()[i];
        std::vector<float> latent{1.0f};
        std::vector<float> clean;
        dense.step(latent, {0.0f}, i, &clean);
        CHECK(std::abs(clean[0] - 1 / std::sqrt(alpha[time])) < 1e-4);
        CHECK(std::abs(latent[0] - std::sqrt(alpha[time - 1] / alpha[time])) < 2e-6);
    }
}

TEST_CASE("DDIM schedules keep every supported step count within the alpha table") {
    for (int steps = 1; steps < 1000; ++steps) {
        GenerateOptions options(SamplingPreset::DdimLeading, 20, 7.5f);
        options.steps = steps;
        REQUIRE_NOTHROW(detail::validate(options));
        detail::DdimScheduler scheduler(steps);
        const auto& times = scheduler.timesteps();
        REQUIRE(times.size() == static_cast<std::size_t>(steps));
        REQUIRE(times.back() == 1);
        REQUIRE(times.front() < 1000);
        for (std::size_t i = 1; i < times.size(); ++i) {
            REQUIRE(times[i - 1] - times[i] == 1000 / steps);
        }
    }
}

TEST_CASE("Guidance preserves the existing CFG formula") {
    std::vector<float> noise{0.1f, -0.2f};
    detail::apply_guidance(noise, {0.3f, 0.4f}, 2.0f);
    CHECK(std::abs(noise[0] - 0.5f) < 1e-6f);
    CHECK(std::abs(noise[1] - 1.0f) < 1e-6f);
    noise = {0.1f};
    detail::apply_guidance(noise, {0.3f}, 0.0f);
    CHECK(noise[0] == 0.1f);
    REQUIRE_THROWS_AS(detail::apply_guidance(noise, {0.3f}, std::numeric_limits<float>::infinity()),
                      std::invalid_argument);
}

TEST_CASE("Denoising preserves latent and preview buffer sizes across steps") {
    auto latent = detail::initial_noise(42);
    const std::vector<float> negative(kLatentElements, 0.1f);
    const std::vector<float> positive(kLatentElements, 0.3f);
    std::vector<float> clean;
    detail::DdimScheduler scheduler(20);
    auto noise = negative;
    detail::apply_guidance(noise, positive, 7.5f);
    for (std::size_t i = 0; i < scheduler.timesteps().size(); ++i) {
        const bool preview = i % 2 == 0;
        scheduler.step(latent, noise, i, preview ? &clean : nullptr);
        REQUIRE(latent.size() == kLatentElements);
        REQUIRE(clean.size() == kLatentElements);
    }
}

TEST_CASE("Seeds preserve explicit values and resolve negative values") {
    GenerateOptions options(SamplingPreset::DdimLeading, 20, 7.5f);
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
    detail::DdimScheduler scheduler(1);
    std::vector<float> latent{1.0f};
    std::vector<float> clean;
    scheduler.step(latent, {0.5f}, 0, &clean);
    REQUIRE(clean.size() == 1);
    const double alpha0 = 1 - 0.00085;
    const double root1 = std::sqrt(0.00085) + (std::sqrt(0.012) - std::sqrt(0.00085)) / 999;
    const double alpha1 = alpha0 * (1 - root1 * root1);
    const double expected = (1 - std::sqrt(1 - alpha1) * 0.5) / std::sqrt(alpha1);
    CHECK(std::abs(clean[0] - expected) < 2e-6);
    CHECK(std::abs(latent[0] - (std::sqrt(alpha0) * expected + std::sqrt(1 - alpha0) * 0.5)) < 2e-6);
}

TEST_CASE("Generation options reject invalid seeds steps and guidance") {
    GenerateOptions options(SamplingPreset::DdimLeading, 20, 7.5f);
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
    detail::DdimScheduler scheduler(20);
    REQUIRE_THROWS_AS(scheduler.step(latent, noise, 0), std::runtime_error);

    std::vector<float> decoded(3 * kImageSize * kImageSize, 0.0f);
    decoded.back() = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(detail::pixels(decoded), std::runtime_error);
}
