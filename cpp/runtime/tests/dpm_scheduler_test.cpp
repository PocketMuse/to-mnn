#include "dpm_scheduler.hpp"
#include "scheduler.hpp"
#include "diffusion.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace sd15;

TEST_CASE("DPM schedules remain finite monotonic and bounded for every supported step count") {
    for (int steps = 1; steps < detail::kTrainSteps; ++steps) {
        detail::DpmScheduler scheduler(steps);
        const auto& times = scheduler.timesteps();
        const auto& sigmas = scheduler.sigmas();
        REQUIRE(times.size() == static_cast<std::size_t>(steps));
        REQUIRE(sigmas.size() == times.size() + 1);
        REQUIRE(sigmas.back() == 0.0f);
        for (std::size_t i = 0; i < times.size(); ++i) {
            REQUIRE(times[i] >= 0);
            REQUIRE(times[i] <= detail::kTrainSteps - 1);
            REQUIRE(std::isfinite(sigmas[i]));
            REQUIRE(sigmas[i] > sigmas[i + 1]);
            if (i > 0) {
                REQUIRE(times[i] <= times[i - 1]);
            }
        }
    }
}

TEST_CASE("DPM scheduler validates requests and rejects nonfinite results") {
    REQUIRE_THROWS_AS(detail::DpmScheduler(0), std::invalid_argument);
    REQUIRE_THROWS_AS(detail::DpmScheduler(1000), std::invalid_argument);
    detail::DpmScheduler scheduler(2);
    std::vector<float> latent{1.0f};
    REQUIRE_THROWS_AS(scheduler.step(latent, {std::numeric_limits<float>::infinity()}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(detail::make_scheduler(static_cast<SamplingPreset>(-1), 20), std::invalid_argument);
    GenerateOptions request(SamplingPreset::Dpmpp2mKarras, 20, 7.5f);
    CHECK(request.sampling == SamplingPreset::Dpmpp2mKarras);
    request.sampling = static_cast<SamplingPreset>(-1);
    REQUIRE_THROWS_AS(detail::validate(request), std::invalid_argument);
    detail::DpmScheduler ordered(2);
    REQUIRE_THROWS_AS(ordered.step(latent, {0.0f}, 1), std::invalid_argument);
}

TEST_CASE("DPM scheduler indexes repeated model timesteps and keeps request state isolated") {
    detail::DpmScheduler with_preview(999);
    auto without_preview = detail::make_scheduler(SamplingPreset::Dpmpp2mKarras, 999);
    std::vector<float> a{0.5f}, b = a, clean;
    for (std::size_t i = 0; i < with_preview.timesteps().size(); ++i) {
        with_preview.step(a, {0.1f}, i, &clean);
        without_preview->step(b, {0.1f}, i);
        REQUIRE(a == b);
        REQUIRE(std::isfinite(a[0]));
    }
    CHECK(a == clean);
}
