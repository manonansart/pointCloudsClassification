cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(visualizer_2)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (visualizer_2 visualizer_2.cpp)
target_link_libraries (visualizer_2 ${PCL_LIBRARIES})
