#include "sd15/converter.hpp"
#include "sd15/conversion_service.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <condition_variable>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using Json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write(const fs::path& path, const std::string& bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(bytes.data(), bytes.size());
}

int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }
    const fs::path root(argv[1]);
    fs::create_directories(root / "template");
    const std::string header = R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    std::string source(8, '\0');
    source[0] = static_cast<char>(header.size());
    source += header;
    source.append("\0\0\x80\x3f\0\0\x80\xbf", 8);
    write(root / "input.safetensors", source);
    write(root / "template/graph.bin", "MNN!");
    Json manifest = {
        {"format", "sd15-mnn-template"}, {"version", 1}, {"mnn_version", "3.6.1"},
        {"template", "graph.bin"}, {"template_size", 4},
        {"files", Json::array({{{"name", "test.mnn"}, {"size", 8},
            {"segments", Json::array({
                {{"kind", "literal"}, {"offset", 0}, {"size", 4}, {"template_offset", 0}},
                {{"kind", "tensor"}, {"offset", 4}, {"size", 4}, {"key", "weight"},
                 {"dtype", "F16"}, {"source_shape", {2}}, {"shape", {2}}, {"positive_zero", false}}
            })}}})}
    };
    const auto save_manifest = [&] { write(root / "template/manifest.json", manifest.dump()); };
    save_manifest();
    sd15::ConvertOptions options{(root / "input.safetensors").string(),
        (root / "template").string(), (root / "out").string(), 4};
    try {
        const auto info = sd15::inspect(options);
        require(info.ok && info.output_bytes == 8 && info.files.size() == 1, "inspect output plan");
        require(!info.hqq_iterations, "float template has no HQQ iterations");
        options.thread_count = 0;
        require(sd15::inspect(options).error.code == sd15::ErrorCode::InvalidInput, "zero threads rejected");
        options.thread_count = sd15::kMaxThreadCount + 1;
        sd15::CancellationToken invalid_threads;
        require(sd15::convert(options, {}, invalid_threads).error.code == sd15::ErrorCode::InvalidInput,
                "excessive threads rejected before writing");
        options.thread_count = 8;
        options.hqq_iterations = -1;
        require(sd15::inspect(options).error.code == sd15::ErrorCode::InvalidInput, "negative iterations rejected");
        options.hqq_iterations = sd15::kHqqIterations + 1;
        sd15::CancellationToken invalid_iterations;
        require(sd15::convert(options, {}, invalid_iterations).error.code == sd15::ErrorCode::InvalidInput,
                "excessive iterations rejected before writing");
        options.hqq_iterations = sd15::kHqqIterations;
        require(!fs::exists(root / "out.partial"), "inspect must not write");

        sd15::CancellationToken token;
        std::vector<sd15::ConvertProgress> events;
        const auto result = sd15::convert(options, [&](const auto& progress) { events.push_back(progress); }, token);
        require(result.status == sd15::ConvertStatus::Succeeded, "conversion succeeds");
        require(!result.hqq_iterations && result.hqq_compute_ms == 0, "float conversion has no HQQ work");
        require(!events.empty() && events.back().stage == sd15::ConvertStage::Completed, "completion event");
        require(events.back().completed_bytes == 8 && events.back().remaining_ms == 0, "completed progress");
        require(result.files.size() == 1 && result.files[0].size_bytes == 8, "output result");
        std::ifstream output(root / "out/test.mnn", std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(output)), {});
        require(bytes == std::string("MNN!\0\x3c\0\xbc", 8), "weight bytes unchanged");
        require(sd15::inspect(options).error.code == sd15::ErrorCode::OutputExists, "structured existing output error");

        options.output_dir = (root / "cancelled").string();
        sd15::CancellationToken cancelled;
        cancelled.request_cancel();
        require(sd15::convert(options, {}, cancelled).status == sd15::ConvertStatus::Cancelled, "pre-cancel");
        require(!fs::exists(root / "cancelled.partial"), "pre-cancel leaves no files");

        sd15::CancellationToken during;
        const auto stopped = sd15::convert(options, [&](const auto& progress) {
            if (progress.stage == sd15::ConvertStage::Converting) {
                during.request_cancel();
            }
        }, during);
        require(stopped.status == sd15::ConvertStatus::Cancelled, "cancel from callback");
        require(!fs::exists(root / "cancelled.partial") && !fs::exists(root / "cancelled"), "cancel cleans staging");

        sd15::CancellationToken before_publish;
        const auto finalized = sd15::convert(options, [&](const auto& progress) {
            if (progress.stage == sd15::ConvertStage::Finalizing) {
                before_publish.request_cancel();
            }
        }, before_publish);
        require(finalized.status == sd15::ConvertStatus::Cancelled, "cancel before publish");
        require(!fs::exists(root / "cancelled.partial") && !fs::exists(root / "cancelled"), "completed files cleaned before publish");

        const auto failed_cleanup = sd15::convert(options, [&](const auto& progress) {
            if (progress.stage == sd15::ConvertStage::Converting) {
                write(root / "cancelled.partial/foreign", "keep");
                throw std::runtime_error("observer failed");
            }
        }, token);
        require(failed_cleanup.status == sd15::ConvertStatus::Failed &&
            failed_cleanup.error.code == sd15::ErrorCode::CleanupFailed, "cleanup failure is explicit");
        require(fs::exists(root / "cancelled.partial/foreign"), "cleanup failure preserves unrelated file");
        fs::remove(root / "cancelled.partial/foreign");
        fs::remove(root / "cancelled.partial");

        sd15::CancellationToken callback_token;
        const auto callback_error = sd15::convert(options, [](const auto&) { throw 42; }, callback_token);
        require(callback_error.error.code == sd15::ErrorCode::CallbackFailed, "callback exception contained");

        manifest["files"][0]["segments"][1]["source_shape"] = {1, 2};
        save_manifest();
        const auto mismatch = sd15::inspect(options);
        require(mismatch.error.code == sd15::ErrorCode::TensorMismatch && mismatch.error.tensor == "weight", "shape error context");
        manifest["files"][0]["segments"][1]["source_shape"] = {2};
        save_manifest();

        write(root / "template/manifest.json", "{");
        const auto malformed = sd15::inspect(options);
        require(malformed.error.code == sd15::ErrorCode::IncompatibleTemplate &&
                malformed.error.path == (root / "template/manifest.json").string(), "manifest error context");
        save_manifest();
        require(sd15::inspect(options).ok, "previous validation error does not affect next inspection");

        options.output_dir = (root / "eta").string();
        auto second = manifest["files"][0];
        second["name"] = "second.mnn";
        manifest["files"].push_back(second);
        save_manifest();
        bool saw_estimate = false;
        sd15::CancellationToken eta_token;
        const auto estimated = sd15::convert(options, [&](const auto& progress) {
            if (progress.stage != sd15::ConvertStage::Completed) {
                require(progress.fraction < 1, "100 percent only after publish");
            }
            if (progress.stage == sd15::ConvertStage::Converting && progress.component == "second") {
                saw_estimate = progress.remaining_ms.has_value();
                require(sd15::cleanup_partial(options).error.code == sd15::ErrorCode::Busy, "cannot clean active job");
                sd15::CancellationToken nested;
                require(sd15::convert(options, {}, nested).error.code == sd15::ErrorCode::Busy, "one conversion per process");
            }
        }, eta_token);
        require(estimated.status == sd15::ConvertStatus::Succeeded && saw_estimate, "ETA after work classes sampled");
        manifest["files"].erase(1);
        save_manifest();
        options.output_dir = (root / "cancelled").string();

        fs::create_directory(root / "cancelled.partial");
        write(root / "cancelled.partial/foreign", "keep");
        require(!sd15::cleanup_partial(options).ok, "refuse unowned staging");
        require(fs::exists(root / "cancelled.partial/foreign"), "preserve foreign files");
        write(root / "cancelled.partial/.sd15-converter.json",
              R"({"format":"sd15-mnn-template","files":["test.mnn"]})");
        write(root / "cancelled.partial/test.mnn", "partial");
        require(sd15::inspect(options).partial_exists, "detect stale output");
        require(sd15::convert(options, {}, eta_token).error.code == sd15::ErrorCode::StaleOutput, "refuse stale output");
        require(!sd15::cleanup_partial(options).ok, "unknown file blocks entire cleanup");
        require(fs::exists(root / "cancelled.partial/test.mnn"), "cleanup validates before deleting");
        fs::remove(root / "cancelled.partial/foreign");
        require(sd15::cleanup_partial(options).ok, "clean owned partial output");
        require(sd15::cleanup_partial(options).ok, "cleanup is idempotent");

        sd15::CancellationToken late;
        const auto committed = sd15::convert(options, [&](const auto& progress) {
            if (progress.stage == sd15::ConvertStage::Completed) {
                late.request_cancel();
                throw std::runtime_error("observer closed");
            }
        }, late);
        require(committed.status == sd15::ConvertStatus::Succeeded, "published result cannot be reversed");

        options.output_dir = (root / "job").string();
        sd15::ConversionService service;
        std::mutex gate;
        std::condition_variable ready;
        bool entered = false;
        bool release = false;
        const auto job = service.start_conversion(options, [&](const auto& snapshot) {
            if (snapshot.result) {
                std::lock_guard<std::mutex> lock(gate);
                ready.notify_all();
                return;
            }
            std::unique_lock<std::mutex> lock(gate);
            entered = true;
            ready.notify_all();
            ready.wait(lock, [&] { return release; });
        });
        require(job.job_id != 0, "job starts");
        {
            std::unique_lock<std::mutex> lock(gate);
            ready.wait(lock, [&] { return entered; });
        }
        require(service.start_conversion(options).error.code == sd15::ErrorCode::Busy, "reject simultaneous start");
        require(service.cancel_conversion(job.job_id), "cancel known job");
        require(service.cancel_conversion(job.job_id), "cancel is idempotent");
        require(service.get_conversion_status(job.job_id)->state == sd15::JobState::Cancelling, "cancelling state");
        require(!service.get_conversion_status(job.job_id + 1), "unknown job");
        {
            std::lock_guard<std::mutex> lock(gate);
            release = true;
        }
        ready.notify_all();
        {
            std::unique_lock<std::mutex> lock(gate);
            ready.wait(lock, [&] { return service.get_conversion_status(job.job_id)->result.has_value(); });
        }
        require(service.get_conversion_status(job.job_id)->state == sd15::JobState::Cancelled, "terminal status retained");
        require(!fs::exists(root / "job.partial"), "job cancellation cleanup");

        const Json recipe = {{"algorithm", "hqq"}, {"bits", 8}, {"iterations", 20},
            {"lp_norm", 0.7}, {"beta", 10.0}, {"group_elements", 2}, {"group_count", 1}};
        const Json common = {{"kind", "quantized"}, {"key", "weight"}, {"source_shape", {2}},
            {"positive_zero", false}, {"quantization", recipe}};
        Json weight = common;
        weight.update({{"field", "Weight"}, {"dtype", "U8"}, {"shape", {2}}, {"offset", 8}, {"size", 2}});
        Json alpha = common;
        alpha.update({{"field", "Alpha"}, {"dtype", "F32"}, {"shape", {1, 2}}, {"offset", 0}, {"size", 8}});
        manifest["version"] = 2;
        manifest["files"] = Json::array({{{"name", "test.mnn"}, {"size", 10},
            {"segments", Json::array({alpha, weight})}}});
        save_manifest();
        options.output_dir = (root / "hqq-zero").string();
        options.hqq_iterations = 0;
        const auto quantized_info = sd15::inspect(options);
        require(quantized_info.ok && quantized_info.hqq_iterations == 0, "inspect reports zero refinement");
        sd15::CancellationToken quantized_token;
        const auto quantized_result = sd15::convert(options, {}, quantized_token);
        require(quantized_result.status == sd15::ConvertStatus::Succeeded && quantized_result.hqq_iterations == 0,
                "conversion reports effective refinement count");
        const auto successful_quantized_bytes = quantized_result.files[0].size_bytes;
        require(successful_quantized_bytes == 10, "HQQ output includes alpha bytes");
        options.output_dir = (root / "hqq-overflow").string();
        std::string overflow_source = source.substr(0, 8 + header.size());
        overflow_source.append("\xff\xff\x7f\xff\xff\xff\x7f\x7f", 8);
        write(root / "input.safetensors", overflow_source);
        const auto overflow = sd15::convert(options, {}, quantized_token);
        require(overflow.status == sd15::ConvertStatus::Failed &&
                overflow.error.code == sd15::ErrorCode::InvalidInput && overflow.error.tensor == "weight" &&
                overflow.error.path == options.checkpoint, "HQQ error gains source context");
        require(overflow.files.empty() && !fs::exists(root / "hqq-overflow.partial"), "HQQ error cleans output");
        std::cout << "Converter API tests passed\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
