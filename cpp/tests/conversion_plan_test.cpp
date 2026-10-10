#include "conversion_plan.hpp"

#include <iostream>
#include <stdexcept>

using namespace sd15;
using namespace sd15::detail;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
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
        require(build_conversion_plan(bytes, 4, source, plan, error), "valid mixed plan");
        require(error.code == ErrorCode::None && error.path.empty(), "success clears old error");
        require(plan.output_bytes == 20 && plan.has_hqq && plan.files.size() == 1, "output metadata");
        const auto& tasks = plan.files[0].tasks;
        require(tasks.size() == 4, "alpha merged into HQQ task");
        require(tasks[0].kind == TaskKind::Hqq && tasks[0].alpha_offset == 0 &&
                tasks[0].output_offset == 8 && tasks[0].group_count == 1 && tasks[0].group_elements == 2,
                "resolved HQQ pair");
        require(tasks[1].kind == TaskKind::Literal && tasks[2].kind == TaskKind::Zero &&
                tasks[3].kind == TaskKind::Float && tasks[3].positive_zero &&
                tasks[3].target_storage == FloatStorage::F16, "task order and storage");

        source.clear();
        bytes.clear();
        json.clear();
        require(tasks[0].tensor == "weight" && tasks[0].source_offset == 123 && tasks[3].elements == 2,
                "plan owns source metadata");
        require(!build_conversion_plan(bytes, 4, source, plan, error), "malformed manifest fails");
        require(error.code == ErrorCode::IncompatibleTemplate, "manifest error classification");
        require(plan.files[0].tasks.size() == 4, "failure preserves previous plan");
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
