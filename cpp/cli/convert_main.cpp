#include "sd15/converter.hpp"

#include <CLI/CLI.hpp>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    sd15::ConvertOptions options;
    CLI::App app{"Convert SD1.5 safetensors using a graph template"};
    const CLI::Validator nonempty(
        [](const std::string& value) { return value.empty() ? "Value must not be empty" : ""; }, "NONEMPTY");
    app.option_defaults()->take_last();
    app.add_option("--checkpoint", options.checkpoint, "Safetensors file")->required()->check(nonempty);
    app.add_option("--template-dir", options.template_dir, "Template directory")->required()->check(nonempty);
    app.add_option("--output", options.output_dir, "New output directory")->required()->check(nonempty);
    app.add_option("--chunk-bytes", options.chunk_bytes, "Input buffer bytes")->capture_default_str();
    app.add_option("--threads", options.thread_count, "HQQ threads (1..8)")->capture_default_str();
    app.add_option("--hqq-iterations", options.hqq_iterations, "HQQ iterations (0..20)")->capture_default_str();
    try {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& error) {
        app.exit(error);
        return error.get_exit_code() == 0 ? 0 : 2;
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
