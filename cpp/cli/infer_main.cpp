#include "sd15/runtime.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <CLI/CLI.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

template <typename Char> int run(int argc, const Char* const* argv) {
    try {
        sd15::RuntimeOptions runtime_options;
        sd15::GenerateOptions options;
        std::string output_name;
        std::string precision = "high";
        std::string tuning = "none";
        CLI::App app{"Generate an SD1.5 image with OpenCL"};
        const CLI::Validator nonempty(
            [](const std::string& value) { return value.empty() ? "Value must not be empty" : ""; }, "NONEMPTY");
        app.option_defaults()->take_last();
        app.add_option("--model-dir", runtime_options.model_dir, "Model directory")->required()->check(nonempty);
        app.add_option("--tokenizer-dir", runtime_options.tokenizer_dir, "Optional tokenizer directory");
        app.add_option("--unet-precision", precision, "UNet compute/storage precision")
            ->check(CLI::IsMember({"high", "normal", "low"}))
            ->capture_default_str();
        app.add_option("--tuning", tuning, "OpenCL tuning for all models")
            ->check(CLI::IsMember({"none", "fast", "normal", "wide"}))
            ->capture_default_str();
        app.add_flag("--unet-low-memory", runtime_options.unet_low_memory, "Use quantized-weight low-memory path");
        app.add_option("--prompt", options.prompt, "Prompt (UTF-8; may be empty)")->required();
        app.add_option("--negative-prompt", options.negative_prompt, "Negative prompt");
        app.add_option("--seed", options.seed, "Negative: random; otherwise uint32 seed")
            ->check(CLI::Range(std::numeric_limits<int64_t>::min(), int64_t{std::numeric_limits<uint32_t>::max()}))
            ->capture_default_str();
        app.add_option("--steps", options.steps, "DDIM steps")->check(CLI::Range(1, 999))->capture_default_str();
        app.add_option("--guidance", options.guidance, "Finite guidance scale")->capture_default_str();
        app.add_option("--output", output_name, "New PNG file")->required()->check(nonempty);
        try {
            app.parse(argc, argv);
        }
        catch (const CLI::ParseError& error) {
            app.exit(error);
            return error.get_exit_code() == 0 ? 0 : 1;
        }
        runtime_options.unet_precision = precision == "high"     ? sd15::Precision::High
                                         : precision == "normal" ? sd15::Precision::Normal
                                                                 : sd15::Precision::Low;
        runtime_options.tuning = tuning == "none"     ? sd15::Tuning::None
                                 : tuning == "fast"   ? sd15::Tuning::Fast
                                 : tuning == "normal" ? sd15::Tuning::Normal
                                                      : sd15::Tuning::Wide;
        if (!std::isfinite(options.guidance)) {
            throw std::invalid_argument("Guidance must be finite");
        }
        const auto output = std::filesystem::u8path(output_name);
        if (output.extension() != ".png") {
            throw std::invalid_argument("Require --output ending in .png");
        }
        if (std::filesystem::exists(output)) {
            throw std::invalid_argument("Output already exists: " + output.u8string());
        }
        const auto started = std::chrono::steady_clock::now();
        sd15::Sd15Runtime runtime(runtime_options);
        const auto image = runtime.generate(options, [](const std::string& stage, int step, int total) {
            std::cout << stage;
            if (step != 0) {
                std::cout << " " << step << "/" << total;
            }
            std::cout << std::endl;
        });
        if (!output.parent_path().empty()) {
            std::filesystem::create_directories(output.parent_path());
        }
        std::ofstream png(output, std::ios::binary);
        if (!png) {
            throw std::runtime_error("Cannot open output: " + output.u8string());
        }
        const auto write = [](void* context, void* data, int size) {
            static_cast<std::ofstream*>(context)->write(static_cast<const char*>(data), size);
        };
        const bool encoded =
            stbi_write_png_to_func(write, &png, image.width, image.height, 3, image.rgb.data(), image.width * 3) != 0;
        png.close();
        if (!encoded || !png) {
            throw std::runtime_error("Failed to write PNG: " + output.u8string());
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "Saved " << output.u8string() << " (" << image.width << "x" << image.height << "), " << elapsed
                  << "s, seed=" << image.seed << ", OpenCL buffer, unet-precision=" << precision
                  << ", tuning=" << tuning << ", unet-low-memory=" << runtime_options.unet_low_memory << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    return run(argc, argv);
}
#else
int main(int argc, char** argv) {
    return run(argc, argv);
}
#endif
