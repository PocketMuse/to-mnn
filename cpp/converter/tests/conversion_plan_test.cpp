#include "conversion_plan.hpp"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

using namespace sd15;
using namespace sd15::detail;

TEST_CASE("Mixed conversion plan owns metadata and survives invalid input") {
    std::string json = R"({
          "format":"sd15-mnn-template", "version":2, "mnn_version":"3.6.1",
          "template":"graph.bin", "template_size":4,
          "files":[{"name":"test.mnn", "size":20, "segments":[
            {"kind":"quantized", "field":"Alpha", "offset":0, "size":8,
             "dtype":"F32", "shape":[1,2], "key":"weight", "source_shape":[2], "positive_zero":false,
             "quantization":{"algorithm":"hqq","bits":8,"iterations":20,"lp_norm":0.7,
               "beta":10.0,"group_elements":2,"group_count":1}},
            {"kind":"quantized", "field":"Weight", "offset":8, "size":2,
             "dtype":"U8", "shape":[2], "key":"weight", "source_shape":[2], "positive_zero":false,
             "quantization":{"algorithm":"hqq","bits":8,"iterations":20,"lp_norm":0.7,
               "beta":10.0,"group_elements":2,"group_count":1}},
            {"kind":"literal", "offset":10,"size":4,"template_offset":0},
            {"kind":"zero", "offset":14,"size":2},
            {"kind":"tensor", "offset":16,"size":4,"dtype":"F16","shape":[2],
             "key":"weight","source_shape":[2],"positive_zero":true}
          ]}]
        })";
    std::vector<uint8_t> bytes(json.begin(), json.end());
    SourceIndex source{{"weight", {123, 2, "F32", {2}}}};
    ConversionPlan plan;
    ConvertError error{ErrorCode::WriteFailed, "old", "old", "old"};
    UNSCOPED_INFO("valid mixed plan");
    REQUIRE(build_conversion_plan(bytes, 4, source, plan, error));
    UNSCOPED_INFO("success clears old error");
    REQUIRE(error.code == ErrorCode::None);
    REQUIRE(error.path.empty());
    UNSCOPED_INFO("output metadata");
    REQUIRE(plan.output_bytes == 20);
    REQUIRE(plan.has_hqq);
    REQUIRE(plan.files.size() == 1);
    const auto& tasks = plan.files[0].tasks;
    UNSCOPED_INFO("alpha merged into HQQ task");
    REQUIRE(tasks.size() == 4);
    UNSCOPED_INFO("resolved HQQ pair");
    REQUIRE(tasks[0].kind == TaskKind::Hqq);
    REQUIRE(tasks[0].alpha_offset == 0);
    REQUIRE(tasks[0].output_offset == 8);
    REQUIRE(tasks[0].group_count == 1);
    REQUIRE(tasks[0].group_elements == 2);
    UNSCOPED_INFO("task order and storage");
    REQUIRE(tasks[1].kind == TaskKind::Literal);
    REQUIRE(tasks[2].kind == TaskKind::Zero);
    REQUIRE(tasks[3].kind == TaskKind::Float);
    REQUIRE(tasks[3].positive_zero);
    REQUIRE(tasks[3].target_storage == FloatStorage::F16);

    source.clear();
    bytes.clear();
    json.clear();
    UNSCOPED_INFO("plan owns source metadata");
    REQUIRE(tasks[0].tensor == "weight");
    REQUIRE(tasks[0].source_offset == 123);
    REQUIRE(tasks[3].elements == 2);
    UNSCOPED_INFO("malformed manifest fails");
    REQUIRE(!build_conversion_plan(bytes, 4, source, plan, error));
    UNSCOPED_INFO("manifest error classification");
    REQUIRE(error.code == ErrorCode::IncompatibleTemplate);
    UNSCOPED_INFO("failure preserves previous plan");
    REQUIRE(plan.files[0].tasks.size() == 4);
}
