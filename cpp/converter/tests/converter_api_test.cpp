#include "sd15/converter.hpp"

#include <filesystem>
#include <fstream>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <thread>
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
    const auto convert_once = [&] {
        const sd15::CancellationToken cancellation;
        return sd15::convert(options, {}, cancellation);
    };
    options.thread_count = 0;
    UNSCOPED_INFO("zero threads rejected");
    REQUIRE(convert_once().error.code == sd15::ErrorCode::InvalidInput);
    options.thread_count = sd15::kMaxThreadCount + 1;
    sd15::CancellationToken invalid_threads;
    UNSCOPED_INFO("excessive threads rejected before writing");
    REQUIRE(sd15::convert(options, {}, invalid_threads).error.code == sd15::ErrorCode::InvalidInput);
    options.thread_count = 8;
    options.hqq_iterations = -1;
    UNSCOPED_INFO("negative iterations rejected");
    REQUIRE(convert_once().error.code == sd15::ErrorCode::InvalidInput);
    options.hqq_iterations = sd15::kHqqIterations + 1;
    sd15::CancellationToken invalid_iterations;
    UNSCOPED_INFO("excessive iterations rejected before writing");
    REQUIRE(sd15::convert(options, {}, invalid_iterations).error.code == sd15::ErrorCode::InvalidInput);
    options.hqq_iterations = sd15::kHqqIterations;
    UNSCOPED_INFO("invalid options must not write");
    REQUIRE(!fs::exists(root / "out"));
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
    REQUIRE(events.back().total_bytes == 8);
    REQUIRE(events.back().remaining_ms == 0);
    UNSCOPED_INFO("output result");
    REQUIRE(result.files.size() == 1);
    REQUIRE(result.files[0].size_bytes == 8);
    REQUIRE(result.files[0].name == "test.mnn");
    std::ifstream output(root / "out/test.mnn", std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(output)), {});
    UNSCOPED_INFO("weight bytes unchanged");
    REQUIRE(bytes == std::string("MNN!\0\x3c\0\xbc", 8));
    UNSCOPED_INFO("structured existing output error");
    REQUIRE(convert_once().error.code == sd15::ErrorCode::OutputExists);

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
    const auto mismatch = convert_once();
    UNSCOPED_INFO("shape error context");
    REQUIRE(mismatch.status == sd15::ConvertStatus::Failed);
    REQUIRE(!fs::exists(root / "cancelled.partial"));
    REQUIRE(mismatch.error.code == sd15::ErrorCode::TensorMismatch);
    REQUIRE(mismatch.error.tensor == "weight");
    manifest["files"][0]["segments"][1]["source_shape"] = {2};
    save_manifest();

    write(root / "template/manifest.json", "{");
    const auto malformed = convert_once();
    UNSCOPED_INFO("manifest error context");
    REQUIRE(malformed.status == sd15::ConvertStatus::Failed);
    REQUIRE(!fs::exists(root / "cancelled.partial"));
    REQUIRE(malformed.error.code == sd15::ErrorCode::IncompatibleTemplate);
    REQUIRE(malformed.error.path == (root / "template/manifest.json").string());
    save_manifest();
    UNSCOPED_INFO("previous validation error does not affect next conversion");
    options.output_dir = (root / "validation-recovered").string();
    REQUIRE(convert_once().status == sd15::ConvertStatus::Succeeded);

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

    options.output_dir = (root / "external-cancel").string();
    manifest["files"].push_back(second);
    save_manifest();
    sd15::CancellationToken external;
    const auto caller = std::this_thread::get_id();
    bool saw_partial_output = false;
    bool callbacks_on_caller = true;
    const auto interrupted = sd15::convert(
        options,
        [&](const auto& progress) {
            callbacks_on_caller = callbacks_on_caller && std::this_thread::get_id() == caller;
            if (progress.stage != sd15::ConvertStage::Converting || progress.component != "second") {
                return;
            }
            saw_partial_output = fs::is_regular_file(root / "external-cancel.partial/test.mnn");
            // 변환 도중 다른 스레드가 취소를 요청한다. 시간 지연에 의존하지 않는다.
            std::thread requester([&] {
                external.request_cancel();
                external.request_cancel();
            });
            requester.join();
        },
        external);
    UNSCOPED_INFO("external cancellation after writing output");
    REQUIRE(saw_partial_output);
    REQUIRE(callbacks_on_caller);
    REQUIRE(interrupted.status == sd15::ConvertStatus::Cancelled);
    REQUIRE(interrupted.error.code == sd15::ErrorCode::None);
    REQUIRE(interrupted.files.empty());
    REQUIRE(!fs::exists(root / "external-cancel.partial"));
    REQUIRE(!fs::exists(root / "external-cancel"));
    sd15::CancellationToken fresh;
    UNSCOPED_INFO("conversion recovers after external cancellation");
    REQUIRE(sd15::convert(options, {}, fresh).status == sd15::ConvertStatus::Succeeded);
    manifest["files"].erase(1);
    save_manifest();

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
