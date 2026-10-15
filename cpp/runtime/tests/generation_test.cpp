#include "sd15/runtime.hpp"
#include "mnn_model_session.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <iostream>

namespace {
std::string model_dir;
std::string tokenizer_dir;
struct Interrupted {};
} // namespace

// 실제 모델·Windows OpenCL을 사용하는 선택적 통합 테스트.
TEST_CASE("Generation preserves requests seeds and recovery") {
    for (auto member : {&sd15::RuntimeOptions::clip, &sd15::RuntimeOptions::unet, &sd15::RuntimeOptions::vae}) {
        sd15::RuntimeOptions invalid_config{model_dir, tokenizer_dir};
        (invalid_config.*member).precision = static_cast<sd15::Precision>(-1);
        REQUIRE_THROWS_AS(sd15::Sd15Runtime(invalid_config), std::invalid_argument);
        (invalid_config.*member).precision = sd15::Precision::High;
        (invalid_config.*member).tuning = static_cast<sd15::Tuning>(-1);
        REQUIRE_THROWS_AS(sd15::Sd15Runtime(invalid_config), std::invalid_argument);
    }
    sd15::RuntimeOptions config{model_dir, tokenizer_dir};
    sd15::Sd15Runtime runtime(config);
    // 생성 후 호출자가 설정을 바꿔도 Runtime의 설정은 유지한다.
    config.unet.precision = static_cast<sd15::Precision>(-1);
    sd15::GenerateOptions request;
    request.prompt = "a photo of a cat";
    request.negative_prompt = "blurry";
    request.steps = 2;
    auto invalid = request;
    invalid.steps = 0;
    REQUIRE_THROWS_AS(runtime.generate(invalid), std::invalid_argument);

    auto original = request;
    std::vector<std::string> stages;
    const auto first = runtime.generate(request, [&](const std::string& stage, int step, int total) {
        stages.push_back(stage + ":" + std::to_string(step));
        CHECK(total == original.steps);
        if (stage == "text_encoder") {
            request.seed = 123;
            bool busy = false;
            try {
                runtime.generate(original);
            }
            catch (const std::runtime_error& error) {
                busy = std::string(error.what()) == "Runtime is busy";
            }
            UNSCOPED_INFO("Reentrant generation accepted");
            REQUIRE(busy);
        }
    });
    const std::vector<std::string> expected_stages{"text_encoder:0", "unet:0", "denoising:1", "denoising:2",
                                                   "vae_decoder:0"};
    CHECK(stages == expected_stages);
    UNSCOPED_INFO("Unexpected image size");
    REQUIRE(first.rgb.size() == 3 * sd15::kImageSize * sd15::kImageSize);
    // 기본 음수 seed로 생성한 결과를 실제 seed로 재현한다.
    original.seed = first.seed;
    REQUIRE_THROWS_AS(runtime.generate(original,
                                       [](const std::string& stage, int, int) {
                                           if (stage == "denoising") {
                                               throw Interrupted{};
                                           }
                                       }),
                      Interrupted);
    const auto repeated = runtime.generate(original, {});
    UNSCOPED_INFO("Actual seed was not preserved");
    CHECK(repeated.seed == first.seed);
    UNSCOPED_INFO("Repeated generation or exception recovery changed image");
    CHECK(first.rgb == repeated.rgb);
    sd15::GenerationCancellationToken cancelled;
    cancelled.request_cancel();
    cancelled.request_cancel();
    bool notified = false;
    REQUIRE_THROWS_AS(runtime.generate(original, cancelled, [&](const std::string&, int, int) { notified = true; }),
                      sd15::GenerationCancelled);
    CHECK_FALSE(notified);
    sd15::GenerationCancellationToken callback_failure;
    REQUIRE_THROWS_AS(runtime.generate(original, callback_failure,
                                       [&](const std::string&, int, int) {
                                           callback_failure.request_cancel();
                                           throw Interrupted{};
                                       }),
                      Interrupted);

    for (const std::string stage_to_cancel : {"text_encoder", "unet", "denoising", "vae_decoder"}) {
        sd15::GenerationCancellationToken cancellation;
        bool reached = false;
        REQUIRE_THROWS_AS(runtime.generate(original, cancellation,
                                           [&](const std::string& stage, int, int) {
                                               CHECK_FALSE(reached);
                                               if (stage == stage_to_cancel) {
                                                   reached = true;
                                                   std::thread requester([&] { cancellation.request_cancel(); });
                                                   requester.join();
                                               }
                                           }),
                          sd15::GenerationCancelled);
        REQUIRE(reached);
    }
    sd15::GenerationCancellationToken fresh;
    CHECK(runtime.generate(original, fresh).rgb == first.rgb);
    request.seed = first.seed == 123 ? 124 : 123;
    const auto changed = runtime.generate(request);
    UNSCOPED_INFO("Explicit seed was not preserved");
    CHECK(changed.seed == request.seed);
    UNSCOPED_INFO("Different seed did not change image");
    CHECK(first.rgb != changed.rgb);
}

// 명시적으로 선택하는 시간 측정. 타이머는 MNN 콜백 내부 중단을 보장하지 않는다.
TEST_CASE("OpenCL cancellation latency experiment", "[.][opencl-cancellation-experiment]") {
    using Clock = std::chrono::steady_clock;
    for (const std::string model : {"unet", "vae_decoder"}) {
        const bool unet = model == "unet";
        const std::vector<int> output_shape = unet ? std::vector<int>{1, 4, 64, 64} : std::vector<int>{1, 3, 512, 512};
        const char* output_name = unet ? "out_sample" : "sample";
        const auto prepare = [&](sd15::detail::MnnModelSession& session) {
            session.input(unet ? "sample" : "latent_sample", {1, 4, 64, 64},
                          std::vector<float>(sd15::kLatentElements, 0.1f));
            if (unet) {
                session.input<int32_t>("timestep", {1}, {500});
                session.input("encoder_hidden_states", {1, 77, 768}, std::vector<float>(77 * 768, 0.1f));
            }
        };
        const auto path = std::filesystem::path(model_dir) / (model + ".mnn");
        std::vector<float> expected;
        {
            sd15::detail::MnnModelSession session(path, unet ? 3 : 1, {});
            prepare(session);
            const auto start = Clock::now();
            expected = session.run(output_name, output_shape);
            std::cout << model
                      << " baseline_ms=" << std::chrono::duration<double, std::milli>(Clock::now() - start).count()
                      << std::endl;
        }
        {
            sd15::GenerationCancellationToken cancellation;
            sd15::detail::MnnModelSession session(path, unet ? 3 : 1, {}, &cancellation);
            prepare(session);
            // 로딩·입력 복사 후 취소를 예약한다. 실제 요청 시점은 스케줄링에 의존한다.
            const auto start = Clock::now();
            std::thread requester([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                cancellation.request_cancel();
            });
            std::exception_ptr failure;
            try {
                session.run(output_name, output_shape);
            }
            catch (...) {
                failure = std::current_exception();
            }
            requester.join();
            std::cout << model
                      << " cancelled_ms=" << std::chrono::duration<double, std::milli>(Clock::now() - start).count()
                      << std::endl;
            REQUIRE(failure);
            REQUIRE_THROWS_AS(std::rethrow_exception(failure), sd15::GenerationCancelled);
        }
        sd15::GenerationCancellationToken fresh;
        sd15::detail::MnnModelSession recovered(path, unet ? 3 : 1, {}, &fresh);
        prepare(recovered);
        CHECK(recovered.run(output_name, output_shape) == expected);
    }
}

int main(int argc, char** argv) {
    Catch::Session session;
    session.cli(
        session.cli() | Catch::Clara::Opt(model_dir, "directory")["--model-dir"]("SD1.5 model directory").required() |
        Catch::Clara::Opt(tokenizer_dir, "directory")["--tokenizer-dir"]("CLIP tokenizer directory").required());
    return session.run(argc, argv);
}
