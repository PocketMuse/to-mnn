cmake_minimum_required(VERSION 3.16)
project(sd15_converter LANGUAGES CXX)

find_package(nlohmann_json 3.11 REQUIRED)
find_package(Threads REQUIRED)
add_library(sd15_converter STATIC src/converter.cpp src/hqq.cpp src/conversion_service.cpp)
target_include_directories(sd15_converter PUBLIC include)
target_link_libraries(sd15_converter PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
target_compile_features(sd15_converter PUBLIC cxx_std_17)
target_compile_definitions(sd15_converter PRIVATE _FILE_OFFSET_BITS=64)
target_compile_options(sd15_converter PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)

add_executable(sd15-convert src/main.cpp)
target_link_libraries(sd15-convert PRIVATE sd15_converter)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(sd15-api-test tests/converter_api_test.cpp)
    target_link_libraries(sd15-api-test PRIVATE sd15_converter nlohmann_json::nlohmann_json Threads::Threads)
    add_test(NAME converter_api COMMAND ${CMAKE_COMMAND} -E env
        SD15_API_TEST=$<TARGET_FILE:sd15-api-test>
        ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/../tests -p test_cpp_api.py)
    add_test(NAME converter COMMAND ${CMAKE_COMMAND} -E env
        SD15_CONVERTER=$<TARGET_FILE:sd15-convert>
        ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/../tests -p test_cpp_converter.py)
endif()
