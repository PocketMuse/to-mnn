#include "sd15/converter.hpp"
#include "sd15/conversion_service.hpp"

#include <filesystem>
#include <fstream>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <condition_variable>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using Json = nlohmann::json;

void write(const fs::path& path, const std::string& bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(bytes.data(), bytes.size());
}

namespace {
std::string work_dir;
}

TEST_CASE("Converter API preserves state cancellation and error contracts") {
    const fs::path root(work_dir);
    fs::create_directories(root / "template");
    const std::string header = R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    std::string source(8, '\0');
    source[0] = static_cast<char>(header.size());
    source += header;
    source.append("\0\0\x80\x3f\0\0\x80\xbf", 8);
    write(root / "input.safetensors", source);
    write(root / "template/graph.bin", "MNN!");
    Json manifest = {
        {"format", "sd15-mnn-template"},
        {"version", 1},
        {"mnn_version", "3.6.1"},
        {"template", "graph.bin"},
        {"template_size", 4},
        {"files", Json::array({{{"name", "test.mnn"},
                                {"size", 8},
                                {"segments",
                                 Json::array({{{"kind", "literal"}, {"offset", 0}, {"size", 4}, {"template_offset", 0}},
                                              {{"kind", "tensor"},
                                               {"offset", 4},
                                               {"size", 4},
                                               {"key", "weight"},
                                               {"dtype", "F16"},
                                               {"source_shape", {2}},
                                               {"shape", {2}},
                                               {"positive_zero", false}}})}}})}};
    const auto save_manifest = [&] { write(root / "template/manifest.json", manifest.dump()); };
    save_manifest();
    sd15::ConvertOptions options{(root / "input.safetensors").string(), (root / "template").string(),
                                 (root / "out").string(), 4};
    const auto info = sd15::inspect(options);
    UNSCOPED_INFO("inspect output plan");
    REQUIRE(info.ok);
    REQUIRE(info.output_bytes == 8);
    REQUIRE(info.files.size() == 1);
    UNSCOPED_INFO("float template has no HQQ iterations");
    REQUIRE(!info.hqq_iterations);
    options.thread_count = 0;
    UNSCOPED_INFO("zero threads rejected");
    REQUIRE(sd15::inspect(options).error.code == sd15::ErrorCode::InvalidInput);
    options.thread_count = sd15::kMaxThreadCount + 1;
    sd15::CancellationToken invalid_threads;
    UNSCOPED_INFO("excessive threads rejected before writing");
    REQUIRE(sd15::convert(options, {}, invalid_threads).error.code == sd15::ErrorCode::InvalidInput);
    options.thread_count = 8;
    options.hqq_iterations = -1;
    UNSCOPED_INFO("negative iterations rejected");
    REQUIRE(sd15::inspect(options).error.code == sd15::ErrorCode::InvalidInput);
    options.hqq_iterations = sd15::kHqqIterations + 1;
    sd15::CancellationToken invalid_iterations;
    UNSCOPED_INFO("excessive iterations rejected before writing");
    REQUIRE(sd15::convert(options, {}, invalid_iterations).error.code == sd15::ErrorCode::InvalidInput);
    options.hqq_iterations = sd15::kHqqIterations;
    UNSCOPED_INFO("inspect must not write");
    REQUIRE(!fs::exists(root / "out.partial"));

    sd15::CancellationToken token;
    std::vector<sd15::ConvertProgress> events;
    const auto result = sd15::convert(
        options, [&](const auto& progress) { events.push_back(progress); }, token);
    UNSCOPED_INFO("conversion succeeds");
    REQUIRE(result.status == sd15::ConvertStatus::Succeeded);
    UNSCOPED_INFO("float conversion has no HQQ work");
    REQUIRE(!result.hqq_iterations);
    REQUIRE(result.hqq_compute_ms == 0);
    UNSCOPED_INFO("completion event");
    REQUIRE(!events.empty());
    REQUIRE(events.back().stage == sd15::ConvertStage::Completed);
    UNSCOPED_INFO("completed progress");
    REQUIRE(events.back().completed_bytes == 8);
    REQUIRE(events.back().remaining_ms == 0);
    UNSCOPED_INFO("output result");
    REQUIRE(result.files.size() == 1);
    REQUIRE(result.files[0].size_bytes == 8);
    std::ifstream output(root / "out/test.mnn", std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(output)), {});
    UNSCOPED_INFO("weight bytes unchanged");
    REQUIRE(bytes == std::string("MNN!\0\x3c\0\xbc", 8));
    UNSCOPED_INFO("structured existing output error");
    REQUIRE(sd15::inspect(options).error.code == sd15::ErrorCode::OutputExists);

    options.output_dir = (root / "cancelled").string();
    sd15::CancellationToken cancelled;
    cancelled.request_cancel();
    UNSCOPED_INFO("pre-cancel");
    REQUIRE(sd15::convert(options, {}, cancelled).status == sd15::ConvertStatus::Cancelled);
    UNSCOPED_INFO("pre-cancel leaves no files");
    REQUIRE(!fs::exists(root / "cancelled.partial"));

    sd15::CancellationToken during;
    const auto stopped = sd15::convert(
        options,
        [&](const auto& progress) {
            if (progress.stage == sd15::ConvertStage::Converting) {
                during.request_cancel();
            }
        },
        during);
    UNSCOPED_INFO("cancel from callback");
    REQUIRE(stopped.status == sd15::ConvertStatus::Cancelled);
    UNSCOPED_INFO("cancel cleans staging");
    REQUIRE(!fs::exists(root / "cancelled.partial"));
    REQUIRE(!fs::exists(root / "cancelled"));

    sd15::CancellationToken before_publish;
    const auto finalized = sd15::convert(
        options,
        [&](const auto& progress) {
            if (progress.stage == sd15::ConvertStage::Finalizing) {
                before_publish.request_cancel();
            }
        },
        before_publish);
    UNSCOPED_INFO("cancel before publish");
    REQUIRE(finalized.status == sd15::ConvertStatus::Cancelled);
    UNSCOPED_INFO("completed files cleaned before publish");
    REQUIRE(!fs::exists(root / "cancelled.partial"));
    REQUIRE(!fs::exists(root / "cancelled"));

    const auto failed_cleanup = sd15::convert(
        options,
        [&](const auto& progress) {
            if (progress.stage == sd15::ConvertStage::Converting) {
                write(root / "cancelled.partial/foreign", "keep");
                throw std::runtime_error("observer failed");
            }
        },
        token);
    UNSCOPED_INFO("cleanup failure is explicit");
    REQUIRE(failed_cleanup.status == sd15::ConvertStatus::Failed);
    REQUIRE(failed_cleanup.error.code == sd15::ErrorCode::CleanupFailed);
    UNSCOPED_INFO("cleanup failure preserves unrelated file");
    REQUIRE(fs::exists(root / "cancelled.partial/foreign"));
    fs::remove(root / "cancelled.partial/foreign");
    fs::remove(root / "cancelled.partial");

    sd15::CancellationToken callback_token;
    const auto callback_error = sd15::convert(
        options, [](const auto&) { throw 42; }, callback_token);
    UNSCOPED_INFO("callback exception contained");
    REQUIRE(callback_error.error.code == sd15::ErrorCode::CallbackFailed);

    manifest["files"][0]["segments"][1]["source_shape"] = {1, 2};
    save_manifest();
    const auto mismatch = sd15::inspect(options);
    UNSCOPED_INFO("shape error context");
    REQUIRE(mismatch.error.code == sd15::ErrorCode::TensorMismatch);
    REQUIRE(mismatch.error.tensor == "weight");
    manifest["files"][0]["segments"][1]["source_shape"] = {2};
    save_manifest();

    write(root / "template/manifest.json", "{");
    const auto malformed = sd15::inspect(options);
    UNSCOPED_INFO("manifest error context");
    REQUIRE(malformed.error.code == sd15::ErrorCode::IncompatibleTemplate);
    REQUIRE(malformed.error.path == (root / "template/manifest.json").string());
    save_manifest();
    UNSCOPED_INFO("previous validation error does not affect next inspection");
    REQUIRE(sd15::inspect(options).ok);

    options.output_dir = (root / "eta").string();
    auto second = manifest["files"][0];
    second["name"] = "second.mnn";
    manifest["files"].push_back(second);
    save_manifest();
    bool saw_estimate = false;
    sd15::CancellationToken eta_token;
    const auto estimated = sd15::convert(
        options,
        [&](const auto& progress) {
            if (progress.stage != sd15::ConvertStage::Completed) {
                UNSCOPED_INFO("100 percent only after publish");
                REQUIRE(progress.fraction < 1);
            }
            if (progress.stage == sd15::ConvertStage::Converting && progress.component == "second") {
                saw_estimate = progress.remaining_ms.has_value();
                UNSCOPED_INFO("cannot clean active job");
                REQUIRE(sd15::cleanup_partial(options).error.code == sd15::ErrorCode::Busy);
                sd15::CancellationToken nested;
                UNSCOPED_INFO("one conversion per process");
                REQUIRE(sd15::convert(options, {}, nested).error.code == sd15::ErrorCode::Busy);
            }
        },
        eta_token);
    UNSCOPED_INFO("ETA after work classes sampled");
    REQUIRE(estimated.status == sd15::ConvertStatus::Succeeded);
    REQUIRE(saw_estimate);
    manifest["files"].erase(1);
    save_manifest();
    options.output_dir = (root / "cancelled").string();

    fs::create_directory(root / "cancelled.partial");
    write(root / "cancelled.partial/foreign", "keep");
    UNSCOPED_INFO("refuse unowned staging");
    REQUIRE(!sd15::cleanup_partial(options).ok);
    UNSCOPED_INFO("preserve foreign files");
    REQUIRE(fs::exists(root / "cancelled.partial/foreign"));
    write(root / "cancelled.partial/.sd15-converter.json", R"({"format":"sd15-mnn-template","files":["test.mnn"]})");
    write(root / "cancelled.partial/test.mnn", "partial");
    UNSCOPED_INFO("detect stale output");
    REQUIRE(sd15::inspect(options).partial_exists);
    UNSCOPED_INFO("refuse stale output");
    REQUIRE(sd15::convert(options, {}, eta_token).error.code == sd15::ErrorCode::StaleOutput);
    UNSCOPED_INFO("unknown file blocks entire cleanup");
    REQUIRE(!sd15::cleanup_partial(options).ok);
    UNSCOPED_INFO("cleanup validates before deleting");
    REQUIRE(fs::exists(root / "cancelled.partial/test.mnn"));
    fs::remove(root / "cancelled.partial/foreign");
    UNSCOPED_INFO("clean owned partial output");
    REQUIRE(sd15::cleanup_partial(options).ok);
    UNSCOPED_INFO("cleanup is idempotent");
    REQUIRE(sd15::cleanup_partial(options).ok);

    sd15::CancellationToken late;
    const auto committed = sd15::convert(
        options,
        [&](const auto& progress) {
            if (progress.stage == sd15::ConvertStage::Completed) {
                late.request_cancel();
                throw std::runtime_error("observer closed");
            }
        },
        late);
    UNSCOPED_INFO("published result cannot be reversed");
    REQUIRE(committed.status == sd15::ConvertStatus::Succeeded);

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
    UNSCOPED_INFO("job starts");
    REQUIRE(job.job_id != 0);
    {
        std::unique_lock<std::mutex> lock(gate);
        ready.wait(lock, [&] { return entered; });
    }
    UNSCOPED_INFO("reject simultaneous start");
    CHECK(service.start_conversion(options).error.code == sd15::ErrorCode::Busy);
    UNSCOPED_INFO("cancel known job");
    CHECK(service.cancel_conversion(job.job_id));
    UNSCOPED_INFO("cancel is idempotent");
    CHECK(service.cancel_conversion(job.job_id));
    UNSCOPED_INFO("cancelling state");
    CHECK(service.get_conversion_status(job.job_id)->state == sd15::JobState::Cancelling);
    UNSCOPED_INFO("unknown job");
    CHECK(!service.get_conversion_status(job.job_id + 1));
    {
        std::lock_guard<std::mutex> lock(gate);
        release = true;
    }
    ready.notify_all();
    {
        std::unique_lock<std::mutex> lock(gate);
        ready.wait(lock, [&] { return service.get_conversion_status(job.job_id)->result.has_value(); });
    }
    UNSCOPED_INFO("terminal status retained");
    REQUIRE(service.get_conversion_status(job.job_id)->state == sd15::JobState::Cancelled);
    UNSCOPED_INFO("job cancellation cleanup");
    REQUIRE(!fs::exists(root / "job.partial"));

    const Json recipe = {{"algorithm", "hqq"}, {"bits", 8},           {"iterations", 20}, {"lp_norm", 0.7},
                         {"beta", 10.0},       {"group_elements", 2}, {"group_count", 1}};
    const Json common = {{"kind", "quantized"},
                         {"key", "weight"},
                         {"source_shape", {2}},
                         {"positive_zero", false},
                         {"quantization", recipe}};
    Json weight = common;
    weight.update({{"field", "Weight"}, {"dtype", "U8"}, {"shape", {2}}, {"offset", 8}, {"size", 2}});
    Json alpha = common;
    alpha.update({{"field", "Alpha"}, {"dtype", "F32"}, {"shape", {1, 2}}, {"offset", 0}, {"size", 8}});
    manifest["version"] = 2;
    manifest["files"] = Json::array({{{"name", "test.mnn"}, {"size", 10}, {"segments", Json::array({alpha, weight})}}});
    save_manifest();
    options.output_dir = (root / "hqq-zero").string();
    options.hqq_iterations = 0;
    const auto quantized_info = sd15::inspect(options);
    UNSCOPED_INFO("inspect reports zero refinement");
    REQUIRE(quantized_info.ok);
    REQUIRE(quantized_info.hqq_iterations == 0);
    sd15::CancellationToken quantized_token;
    const auto quantized_result = sd15::convert(options, {}, quantized_token);
    UNSCOPED_INFO("conversion reports effective refinement count");
    REQUIRE(quantized_result.status == sd15::ConvertStatus::Succeeded);
    REQUIRE(quantized_result.hqq_iterations == 0);
    const auto successful_quantized_bytes = quantized_result.files[0].size_bytes;
    UNSCOPED_INFO("HQQ output includes alpha bytes");
    REQUIRE(successful_quantized_bytes == 10);
    options.output_dir = (root / "hqq-overflow").string();
    std::string overflow_source = source.substr(0, 8 + header.size());
    overflow_source.append("\xff\xff\x7f\xff\xff\xff\x7f\x7f", 8);
    write(root / "input.safetensors", overflow_source);
    const auto overflow = sd15::convert(options, {}, quantized_token);
    UNSCOPED_INFO("HQQ error gains source context");
    REQUIRE(overflow.status == sd15::ConvertStatus::Failed);
    REQUIRE(overflow.error.code == sd15::ErrorCode::InvalidInput);
    REQUIRE(overflow.error.tensor == "weight");
    REQUIRE(overflow.error.path == options.checkpoint);
    UNSCOPED_INFO("HQQ error cleans output");
    REQUIRE(overflow.files.empty());
    REQUIRE(!fs::exists(root / "hqq-overflow.partial"));
}

int main(int argc, char** argv) {
    Catch::Session session;
    session.cli(session.cli() |
                Catch::Clara::Opt(work_dir, "directory")["--work-dir"]("Temporary test directory").required());
    return session.run(argc, argv);
}
