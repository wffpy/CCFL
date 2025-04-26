cmake_minimum_required(VERSION 3.10)
project(CCF LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
# find_package(NCCL REQUIRED)

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB TEST_FILES ${PROJECT_SOURCE_DIR}/test/*.cu ${PROJECT_SOURCE_DIR}/test/*.cpp)

# 检查NCCL_INCLUDE_DIR和NCCL_LIB_DIR环境变量，否则设置默认值
if(NOT DEFINED ENV{NCCL_INCLUDE_DIR} OR "$ENV{NCCL_INCLUDE_DIR}" STREQUAL "")
    set(NCCL_INCLUDE_DIR "/usr/include")
else()
    set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR})
endif()

if(NOT DEFINED ENV{NCCL_LIB_DIR} OR "$ENV{NCCL_LIB_DIR}" STREQUAL "")
    set(NCCL_LIB_DIR "/usr/lib/x86_64-linux-gnu")
else()
    set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR})
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${NCCL_INCLUDE_DIR})
link_directories(${NCCL_LIB_DIR})

add_library(ccfl ${SRC_FILES})

target_link_libraries(ccfl PRIVATE ${CUDA_LIBRARIES} nccl) 