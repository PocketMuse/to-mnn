cmake_minimum_required(VERSION 3.19)
project(sd15_pymnn LANGUAGES CXX)

if(NOT MSVC)
    message(FATAL_ERROR "This binding uses the Windows MSVC MNN Runtime")
endif()

find_package(Python3 3.12...<3.13 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES
    IMPORTED_LOCATION "${MNN_RUNTIME_DIR}/MNN.dll"
    IMPORTED_IMPLIB "${MNN_RUNTIME_DIR}/MNN.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${MNN_SOURCE_DIR}/include")

# 공식 바인딩의 추론 API만 빌드한다.
Python3_add_library(_mnncengine MODULE WITH_SOABI "${MNN_SOURCE_DIR}/pymnn/src/MNN.cc")
target_link_libraries(_mnncengine PRIVATE MNN Python3::NumPy)
target_include_directories(_mnncengine PRIVATE
    "${MNN_SOURCE_DIR}/source"
    "${MNN_SOURCE_DIR}/express"
    "${MNN_SOURCE_DIR}/express/module"
    "${MNN_SOURCE_DIR}/schema/current"
    "${MNN_SOURCE_DIR}/3rd_party/flatbuffers/include")
target_compile_features(_mnncengine PRIVATE cxx_std_17)
target_compile_definitions(_mnncengine PRIVATE
    PYMNN_EXPR_API PYMNN_NUMPY_USABLE USING_MNN_DLL NOMINMAX)
target_compile_options(_mnncengine PRIVATE /bigobj /Zc:preprocessor /utf-8)
