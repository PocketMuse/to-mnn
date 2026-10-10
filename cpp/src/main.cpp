#include "sd15/converter.hpp"

#include <charconv>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    sd15::ConvertOptions options;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Expected a value for %s\n", argv[i]);
            return 2;
        }
        const std::string flag = argv[i];
        const std::string value = argv[i + 1];
        if (flag == "--checkpoint") {
            options.checkpoint = value;
        }
        else if (flag == "--template-dir") {
            options.template_dir = value;
        }
        else if (flag == "--output") {
            options.output_dir = value;
        }
        else if (flag == "--chunk-bytes") {
            const auto result = std::from_chars(value.data(), value.data() + value.size(), options.chunk_bytes);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
                std::fprintf(stderr, "Invalid chunk size\n");
                return 2;
            }
        }
        else if (flag == "--threads") {
            const auto result = std::from_chars(value.data(), value.data() + value.size(), options.thread_count);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
                std::fprintf(stderr, "Invalid thread count\n");
                return 2;
            }
        }
        else if (flag == "--hqq-iterations") {
            const auto result = std::from_chars(value.data(), value.data() + value.size(), options.hqq_iterations);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
                std::fprintf(stderr, "Invalid HQQ iterations\n");
                return 2;
            }
        }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    if (options.checkpoint.empty() || options.template_dir.empty() || options.output_dir.empty()) {
        std::fprintf(
            stderr,
            "Usage: sd15-convert --checkpoint FILE --template-dir DIR --output DIR [--chunk-bytes N] [--hqq-iterations N] [--threads N]\n");
        return 2;
    }
    sd15::CancellationToken cancellation;
    const auto result = sd15::convert(options, {}, cancellation);
    if (result.status != sd15::ConvertStatus::Succeeded) {
        std::fprintf(stderr, "%s: %s\n", sd15::error_code_name(result.error.code), result.error.message.c_str());
        return 1;
    }
    std::printf("Saved %s (chunk buffer: %zu bytes)\n", options.output_dir.c_str(), options.chunk_bytes);
    std::printf("elapsed_ms=%llu hqq_compute_ms=%.3f hqq_iterations=%s thread_count=%u\n",
                static_cast<unsigned long long>(result.elapsed_ms), result.hqq_compute_ms,
                result.hqq_iterations ? std::to_string(*result.hqq_iterations).c_str() : "none", options.thread_count);
    return 0;
}
