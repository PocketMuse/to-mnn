#include "sd15/runtime.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <stdexcept>

namespace {
std::string model_dir;
std::string tokenizer_dir;
struct Interrupted {};
} // namespace

// 실제 모델·Windows OpenCL을 사용하는 선택적 통합 테스트.
TEST_CASE("Generation preserves requests seeds and recovery") {
    sd15::RuntimeOptions invalid_config{model_dir, tokenizer_dir};
    invalid_config.unet_precision = static_cast<sd15::Precision>(-1);
    REQUIRE_THROWS_AS(sd15::Sd15Runtime(invalid_config), std::invalid_argument);
    invalid_config.unet_precision = sd15::Precision::High;
    invalid_config.tuning = static_cast<sd15::Tuning>(-1);
    REQUIRE_THROWS_AS(sd15::Sd15Runtime(invalid_config), std::invalid_argument);
    sd15::RuntimeOptions config{model_dir, tokenizer_dir};
    sd15::Sd15Runtime runtime(config);
    // 생성 후 호출자가 설정을 바꿔도 Runtime의 설정은 유지한다.
    config.unet_precision = static_cast<sd15::Precision>(-1);
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
    const auto repeated = runtime.generate(original);
    UNSCOPED_INFO("Actual seed was not preserved");
    CHECK(repeated.seed == first.seed);
    UNSCOPED_INFO("Repeated generation or exception recovery changed image");
    CHECK(first.rgb == repeated.rgb);
    request.seed = first.seed == 123 ? 124 : 123;
    const auto changed = runtime.generate(request);
    UNSCOPED_INFO("Explicit seed was not preserved");
    CHECK(changed.seed == request.seed);
    UNSCOPED_INFO("Different seed did not change image");
    CHECK(first.rgb != changed.rgb);
}

int main(int argc, char** argv) {
    Catch::Session session;
    session.cli(
        session.cli() | Catch::Clara::Opt(model_dir, "directory")["--model-dir"]("SD1.5 model directory").required() |
        Catch::Clara::Opt(tokenizer_dir, "directory")["--tokenizer-dir"]("CLIP tokenizer directory").required());
    return session.run(argc, argv);
}
