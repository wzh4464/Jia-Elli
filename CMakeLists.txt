cmake_minimum_required(VERSION 3.10) # 指定CMake的最低版本要求

project(EllipseDetector) # 项目名称

# set debug
set(CMAKE_BUILD_TYPE "Debug")

set(OpenCV_DIR "/home/zihan/opencv2_install/share/OpenCV/") # OpenCV的安装路径

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# -fpermissive
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 包含OpenCV的头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# ├── CMakeLists.txt
# ├── code
# │   ├── CNEllipseDetector.cpp
# │   ├── CNEllipseDetector.h
# │   ├── common.cpp
# │   ├── common.h
# │   ├── dirent.h
# │   ├── Main.cpp
# │   ├── stdafx.h
# │   ├── tools.cpp
# │   └── tools.h

# output version information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# 指定头文件目录，这样在include时不需要写完整路径
include_directories(${PROJECT_SOURCE_DIR}/code)

# source directory
aux_source_directory(${PROJECT_SOURCE_DIR}/code SRC_LIST)

# 添加要编译的源文件
add_executable(${PROJECT_NAME} ${SRC_LIST})

# 链接OpenCV库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
