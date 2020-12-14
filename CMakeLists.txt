cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(deepfake)

list(APPEND CMAKE_PREFIX_PATH "/home/mnuppnau/deepfake/external/libtorch")
list(APPEND CMAKE_PREFIX_PATH "/usr/local/include/opencv4/opencv2")

find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_VERBOSE_MAKEFILE on)

#include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(deepfake src/deepfake/main.cpp src/deepfake/main.h)

target_link_libraries(deepfake "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
#target_link_libraries(deepfake "${OpenCV_LIBS}")

message(STATUS "${OpenCV_LIBS}")

set_property(TARGET deepfake PROPERTY CXX_STANDARD 14)
