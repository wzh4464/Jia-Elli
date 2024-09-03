cmake_minimum_required(VERSION 3.10)
project(EllipseDetector)
set(OpenCV_DIR "/home/zihan/opencv_builds/opencv-2.4.13.7/share/OpenCV/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -g") # 添加 -g 标志来保留调试符号
set(CMAKE_BUILD_TYPE Debug) # 设置构建类型为 Debug

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
include_directories(${PROJECT_SOURCE_DIR}/code)
aux_source_directory(${PROJECT_SOURCE_DIR}/code SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Demo)