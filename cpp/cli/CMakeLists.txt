find_package(CLI11 CONFIG REQUIRED)

if(SD15_BUILD_CONVERTER)
    add_executable(sd15-convert convert_main.cpp)
    target_link_libraries(sd15-convert PRIVATE sd15_converter CLI11::CLI11)

    if(BUILD_TESTING)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        add_test(NAME converter COMMAND ${CMAKE_COMMAND} -E env
            SD15_CONVERTER=$<TARGET_FILE:sd15-convert>
            ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/../../tests -p test_cpp_converter.py)
    endif()
endif()

if(SD15_BUILD_RUNTIME)
    add_executable(sd15-infer infer_main.cpp)
    target_link_libraries(sd15-infer PRIVATE sd15_runtime CLI11::CLI11)
    target_include_directories(sd15-infer SYSTEM PRIVATE
        "${MNN_SOURCE_DIR}/3rd_party/imageHelper")
    if(MSVC)
        target_compile_options(sd15-infer PRIVATE /utf-8 /Zc:preprocessor)
        target_compile_definitions(sd15-infer PRIVATE NOMINMAX)
        add_custom_command(TARGET sd15-infer POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:sd15_mnn>" "$<TARGET_FILE_DIR:sd15-infer>")
    endif()
endif()
