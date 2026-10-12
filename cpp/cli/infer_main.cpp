#include "sd15/runtime.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace {

int64_t seed_option(const std::string& value) {
    int64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        result > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("seed must be negative or in [0, 4294967295]: " + value);
    }
    return result;
}

uint32_t unsigned_option(const std::string& value) {
    uint32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("Expected unsigned 32-bit integer: " + value);
    }
    return result;
}

int run(const std::vector<std::string>& args) {
    try {
        sd15::RuntimeOptions runtime_options;
        sd15::GenerateOptions options;
        std::filesystem::path output;
        bool has_prompt = false;
        for (std::size_t i = 1; i < args.size(); ++i) {
            const auto& flag = args[i];
            if (flag == "--help") {
                std::cout << "Usage: sd15-infer --model-dir DIR --prompt TEXT --output image.png\n"
                             "  [--tokenizer-dir DIR] [--negative-prompt TEXT] [--seed INT64]\n"
                             "  [--steps 1..999] [--guidance FLOAT]\n"
                             "  seed: default -1; negative selects a random seed, otherwise 0..4294967295\n";
                return 0;
            }
            if (i + 1 == args.size()) {
                throw std::invalid_argument("Missing value: " + flag);
            }
            const auto& value = args[++i];
            if (flag == "--model-dir") {
                runtime_options.model_dir = value;
            }
            else if (flag == "--tokenizer-dir") {
                runtime_options.tokenizer_dir = value;
            }
            else if (flag == "--prompt") {
                options.prompt = value;
                has_prompt = true;
            }
            else if (flag == "--negative-prompt") {
                options.negative_prompt = value;
            }
            else if (flag == "--seed") {
                options.seed = seed_option(value);
            }
            else if (flag == "--steps") {
                const auto steps = unsigned_option(value);
                if (steps < 1 || steps > 999) {
                    throw std::invalid_argument("steps must be in [1, 999]");
                }
                options.steps = static_cast<int>(steps);
            }
            else if (flag == "--guidance") {
                std::size_t count = 0;
                options.guidance = std::stof(value, &count);
                if (count != value.size() || !std::isfinite(options.guidance)) {
                    throw std::invalid_argument("Guidance must be finite");
                }
            }
            else if (flag == "--output") {
                output = std::filesystem::u8path(value);
            }
            else {
                throw std::invalid_argument("Unknown option: " + flag);
            }
        }
        if (runtime_options.model_dir.empty() || !has_prompt || output.empty() || output.extension() != ".png") {
            throw std::invalid_argument("Require --model-dir, --prompt and --output ending in .png");
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
                  << "s, seed=" << image.seed << ", OpenCL high/buffer/tuning-none\n";
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
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, nullptr, 0, nullptr, nullptr);
        if (size == 0) {
            return 1;
        }
        std::string value(size, '\0');
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, value.data(), size, nullptr, nullptr);
        value.pop_back();
        args.push_back(std::move(value));
    }
    return run(args);
}
#else
int main(int argc, char** argv) {
    return run({argv, argv + argc});
}
#endif
