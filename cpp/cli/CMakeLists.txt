add_executable(sd15-convert convert_main.cpp)
target_link_libraries(sd15-convert PRIVATE sd15_converter)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME converter COMMAND ${CMAKE_COMMAND} -E env
        SD15_CONVERTER=$<TARGET_FILE:sd15-convert>
        ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/../../tests -p test_cpp_converter.py)
endif()
