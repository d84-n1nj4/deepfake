cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(Cassava)

list(APPEND CMAKE_PREFIX_PATH "/home/mnuppnau/cassava/external/libtorch")
list(APPEND CMAKE_PREFIX_PATH "/home/mnuppnau/cassava/external/opencv-master/build")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_VERBOSE_MAKEFILE on)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(cassava src/cassava/datasets/base_dataset.cpp)
target_link_libraries(cassava "${TORCH_LIBRARIES}")
target_link_libraries(cassava "${OpenCV_LIBS}")
set_property(TARGET cassava PROPERTY CXX_STANDARD 14)
