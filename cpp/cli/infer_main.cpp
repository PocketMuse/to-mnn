#include "sd15/runtime.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <CLI/CLI.hpp>
#include <array>
#include <map>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void save_png(const std::filesystem::path& output, const sd15::Image& image) {
    if (std::filesystem::exists(output)) {
        throw std::invalid_argument("Output already exists: " + output.u8string());
    }
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
}

template <typename Char> int run(int argc, const Char* const* argv) {
    try {
        sd15::RuntimeOptions runtime_options;
        sd15::GenerateOptions options;
        std::string output_name;
        std::string preview_directory;
        const std::array<std::string, 3> names{"clip", "unet", "vae"};
        std::array<std::string, 3> precisions{"high", "high", "high"};
        std::array<std::string, 3> tunings{"none", "none", "none"};
        const std::array<sd15::ModelOptions*, 3> models{&runtime_options.clip, &runtime_options.unet,
                                                        &runtime_options.vae};
        const std::map<std::string, sd15::Precision> precision_values{
            {"high", sd15::Precision::High}, {"normal", sd15::Precision::Normal}, {"low", sd15::Precision::Low}};
        const std::map<std::string, sd15::Tuning> tuning_values{{"none", sd15::Tuning::None},
                                                                {"fast", sd15::Tuning::Fast},
                                                                {"normal", sd15::Tuning::Normal},
                                                                {"wide", sd15::Tuning::Wide}};
        std::string tuning = "none";
        CLI::App app{"Generate an SD1.5 image with OpenCL"};
        const CLI::Validator nonempty(
            [](const std::string& value) { return value.empty() ? "Value must not be empty" : ""; }, "NONEMPTY");
        app.option_defaults()->take_last();
        app.add_option("--model-dir", runtime_options.model_dir, "Model directory")->required()->check(nonempty);
        app.add_option("--tokenizer-dir", runtime_options.tokenizer_dir, "Optional tokenizer directory");
        app.add_option("--tuning", tuning, "Default OpenCL tuning for all models")
            ->check(CLI::IsMember(tuning_values))
            ->capture_default_str();
        for (std::size_t i = 0; i < models.size(); ++i) {
            const auto prefix = "--" + names[i];
            app.add_option(prefix + "-precision", precisions[i], "Requested compute/storage precision")
                ->check(CLI::IsMember(precision_values))
                ->capture_default_str();
            app.add_option(prefix + "-tuning", tunings[i], "Override --tuning for this model")
                ->check(CLI::IsMember(tuning_values));
            app.add_flag(prefix + "-low-memory", models[i]->low_memory, "Use quantized-weight low-memory path");
        }
        app.add_option("--prompt", options.prompt, "Prompt (UTF-8; may be empty)")->required();
        app.add_option("--negative-prompt", options.negative_prompt, "Negative prompt");
        app.add_option("--seed", options.seed, "Negative: random; otherwise uint32 seed")
            ->check(CLI::Range(std::numeric_limits<int64_t>::min(), int64_t{std::numeric_limits<uint32_t>::max()}))
            ->capture_default_str();
        app.add_option("--steps", options.steps, "DDIM steps")->check(CLI::Range(1, 999))->capture_default_str();
        app.add_option("--guidance", options.guidance, "Finite guidance scale")->capture_default_str();
        app.add_option("--output", output_name, "New PNG file")->required()->check(nonempty);
        app.add_option("--preview-decoder", runtime_options.preview_decoder_path, "Local TAESD MNN decoder");
        app.add_option("--preview-interval", options.preview_interval, "Preview every N steps; 0 disables")
            ->check(CLI::Range(0, 999))
            ->capture_default_str();
        app.add_option("--preview-dir", preview_directory, "New directory for step PNGs");
        try {
            app.parse(argc, argv);
        }
        catch (const CLI::ParseError& error) {
            app.exit(error);
            return error.get_exit_code() == 0 ? 0 : 1;
        }
        for (std::size_t i = 0; i < models.size(); ++i) {
            models[i]->precision = precision_values.at(precisions[i]);
            if (app.count("--" + names[i] + "-tuning") == 0) {
                tunings[i] = tuning;
            }
            models[i]->tuning = tuning_values.at(tunings[i]);
        }
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
        sd15::InferencePreview preview;
        if (options.preview_interval > 0) {
            if (preview_directory.empty() || std::filesystem::exists(std::filesystem::u8path(preview_directory))) {
                throw std::invalid_argument("Previews require a new --preview-dir");
            }
            preview = [&](int step, int, const sd15::Image& image) {
                const auto name = "step-" + std::to_string(step) + ".png";
                const auto path = std::filesystem::u8path(preview_directory) / name;
                save_png(path, image);
                std::cout << "Preview " << step << ": " << path.u8string() << std::endl;
            };
        }
        for (std::size_t i = 0; i < models.size(); ++i) {
            std::cout << names[i] << " requested: precision=" << precisions[i] << ", tuning=" << tunings[i]
                      << ", low-memory=" << models[i]->low_memory << std::endl;
        }
        const auto started = std::chrono::steady_clock::now();
        sd15::Sd15Runtime runtime(runtime_options);
        const auto image = runtime.generate(
            options,
            [](const std::string& stage, int step, int total) {
                std::cout << stage;
                if (step != 0) {
                    std::cout << " " << step << "/" << total;
                }
                std::cout << std::endl;
            },
            preview);
        save_png(output, image);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "Saved " << output.u8string() << " (" << image.width << "x" << image.height << "), " << elapsed
                  << "s, seed=" << image.seed << ", OpenCL buffer\n";
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
