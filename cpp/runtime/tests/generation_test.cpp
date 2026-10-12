#include "sd15/runtime.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}
struct Interrupted {};
} // namespace

// 실제 모델·Windows OpenCL을 사용하는 선택적 통합 테스트.
int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument("Expected model and tokenizer directories");
        }
        sd15::Sd15Runtime runtime({argv[1], argv[2]});
        sd15::GenerateOptions request;
        request.prompt = "a photo of a cat";
        request.negative_prompt = "blurry";
        request.steps = 2;
        auto invalid = request;
        invalid.steps = 0;
        bool rejected = false;
        try {
            runtime.generate(invalid);
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "Invalid generation accepted");

        auto original = request;
        const auto first = runtime.generate(request, [&](const std::string& stage, int, int) {
            if (stage == "text_encoder") {
                request.seed = 123;
                bool busy = false;
                try {
                    runtime.generate(original);
                }
                catch (const std::runtime_error& error) {
                    busy = std::string(error.what()) == "Runtime is busy";
                }
                require(busy, "Reentrant generation accepted");
            }
        });
        require(first.rgb.size() == 3 * sd15::kImageSize * sd15::kImageSize, "Unexpected image size");
        // 기본 음수 seed로 생성한 결과를 실제 seed로 재현한다.
        original.seed = first.seed;
        bool interrupted = false;
        try {
            runtime.generate(original, [](const std::string& stage, int, int) {
                if (stage == "denoising") {
                    throw Interrupted{};
                }
            });
        }
        catch (const Interrupted&) {
            interrupted = true;
        }
        require(interrupted, "Callback exception was lost");
        const auto repeated = runtime.generate(original);
        require(repeated.seed == first.seed, "Actual seed was not preserved");
        require(first.rgb == repeated.rgb, "Repeated generation or exception recovery changed image");
        request.seed = first.seed == 123 ? 124 : 123;
        const auto changed = runtime.generate(request);
        require(changed.seed == request.seed, "Explicit seed was not preserved");
        require(first.rgb != changed.rgb, "Different seed did not change image");
        std::cout << "Repeated generation, request snapshot, reentry and failure recovery passed\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
