cmake_minimum_required(VERSION 3.18)
project(threaded_engine C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

# whatever source file you wish to build, set it here
set(SOURCE threaded_engine_interface.cc)

add_subdirectory(mlc-llm)

add_executable(${PROJECT_NAME} ${SOURCE})
add_dependencies(${PROJECT_NAME} mlc_llm)

# includes
set(TVM_SOURCE_DIR mlc-llm/3rdparty/tvm)
target_include_directories(${PROJECT_NAME} PRIVATE mlc-llm/cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
    ${TVM_SOURCE_DIR}/include
    ${TVM_SOURCE_DIR}/3rdparty/dlpack/include
    ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
    ${TVM_SOURCE_DIR}/3rdparty/picojson
    ${TVM_SOURCE_DIR}/include/tvm/runtime
)
target_include_directories(${PROJECT_NAME} PRIVATE mlc-llm/3rdparty/stb)
target_include_directories(${PROJECT_NAME} PRIVATE mlc-llm/3rdparty/tokenizers-cpp/include)
target_include_directories(${PROJECT_NAME} PRIVATE mlc-llm/3rdparty/xgrammar/include)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        mlc_llm
)

target_link_libraries(mlc_llm
    PUBLIC
        ${CMAKE_SOURCE_DIR}/libdmlc.a
)