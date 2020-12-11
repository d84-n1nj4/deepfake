cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(deepfake)

list(APPEND CMAKE_PREFIX_PATH "/home/mnuppnau/deepfake/external/libtorch")
list(APPEND CMAKE_PREFIX_PATH "/home/mnuppnau/deepfake/external/opencv-master/build")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_VERBOSE_MAKEFILE on)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(deepfake src/deepfake/datasets/base_dataset.cpp)
target_link_libraries(deepfake "${TORCH_LIBRARIES}")
target_link_libraries(deepfake "${OpenCV_LIBS}")
set_property(TARGET deepfake PROPERTY CXX_STANDARD 14)
