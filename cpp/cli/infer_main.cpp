#include "sd15/runtime.hpp"

#include <json.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

using Json = nlohmann::json;
constexpr uintmax_t kMaxInputBytes = 2 * 1024 * 1024;

sd15::GenerateOptions read_input(const std::filesystem::path& path) {
    if (std::filesystem::file_size(path) > kMaxInputBytes) {
        throw std::invalid_argument("Input JSON exceeds 2 MiB");
    }
    std::ifstream stream(path);
    const auto input = Json::parse(stream);
    if (input.at("format") != "sd15-inference-input" || input.at("version") != 1) {
        throw std::invalid_argument("Unsupported inference input");
    }
    sd15::GenerateOptions options;
    for (const char* name : {"tokens", "negative_tokens"}) {
        const auto& values = input.at(name);
        if (!values.is_array() || values.size() != sd15::kSequenceLength) {
            throw std::invalid_argument("Expected 77 token IDs");
        }
        for (const auto& value : values) {
            if (!value.is_number_integer() || value.get<double>() < 0 || value.get<double>() >= sd15::kVocabularySize) {
                throw std::invalid_argument("Invalid token ID");
            }
        }
    }
    options.tokens = input.at("tokens").get<std::vector<int32_t>>();
    options.negative_tokens = input.at("negative_tokens").get<std::vector<int32_t>>();
    options.initial_latent = input.at("initial_latent").get<std::vector<float>>();
    if (!input.at("steps").is_number_integer()) {
        throw std::invalid_argument("steps must be an integer");
    }
    const auto steps = input.at("steps").get<int64_t>();
    if (steps < 1 || steps > 999) {
        throw std::invalid_argument("steps must be in [1, 999]");
    }
    options.steps = static_cast<int>(steps);
    options.guidance = input.at("guidance").get<float>();
    options.vae_scaling_factor = input.at("vae_scaling_factor").get<float>();
    const auto& config = input.at("scheduler");
    // 첫 버전은 기존 SD1.5 runner의 고정 스케줄만 지원한다.
    if (config.at("beta_schedule") != "scaled_linear" || config.at("prediction_type") != "epsilon" ||
        config.at("clip_sample") != false || config.at("num_train_timesteps") != 1000 ||
        config.at("steps_offset") != 1 || config.at("set_alpha_to_one") != false ||
        config.at("beta_start") != 0.00085 || config.at("beta_end") != 0.012) {
        throw std::invalid_argument("Expected SD1.5 scaled-linear epsilon/DDIM schedule");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string model_dir;
        std::filesystem::path input;
        std::filesystem::path output;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--help") {
                std::cout << "Usage: sd15-infer --model-dir DIR --input request.json --output image.png\n";
                return 0;
            }
            if (i + 1 == argc) {
                throw std::invalid_argument("Missing value: " + flag);
            }
            const char* value = argv[++i];
            if (flag == "--model-dir") {
                model_dir = value;
            }
            else if (flag == "--input") {
                input = value;
            }
            else if (flag == "--output") {
                output = value;
            }
            else {
                throw std::invalid_argument("Unknown option: " + flag);
            }
        }
        if (model_dir.empty() || input.empty() || output.empty() || output.extension() != ".png") {
            throw std::invalid_argument("Require --model-dir, --input and --output ending in .png");
        }
        if (std::filesystem::exists(output)) {
            throw std::invalid_argument("Output already exists: " + output.string());
        }
        auto options = read_input(input);
        options.model_dir = model_dir;
        if (!output.parent_path().empty()) {
            std::filesystem::create_directories(output.parent_path());
        }
        const auto started = std::chrono::steady_clock::now();
        const auto image = sd15::generate(options, [](const std::string& stage, int step, int total) {
            std::cout << stage;
            if (step != 0) {
                std::cout << " " << step << "/" << total;
            }
            std::cout << std::endl;
        });
        std::ofstream png(output, std::ios::binary);
        if (!png) {
            throw std::runtime_error("Cannot open output: " + output.string());
        }
        const auto write = [](void* context, void* data, int size) {
            static_cast<std::ofstream*>(context)->write(static_cast<const char*>(data), size);
        };
        const bool encoded =
            stbi_write_png_to_func(write, &png, image.width, image.height, 3, image.rgb.data(), image.width * 3) != 0;
        png.close();
        if (!encoded || !png) {
            throw std::runtime_error("Failed to write PNG: " + output.string());
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "Saved " << output.string() << " (" << image.width << "x" << image.height << "), " << elapsed
                  << "s, OpenCL high/buffer/tuning-none\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
