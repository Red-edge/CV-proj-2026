cmake_minimum_required(VERSION 3.22)
project(CVProj2026Cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CVPROJ_ENABLE_HIKROBOT_MVS "Enable Hikrobot MVS SDK integration" ON)
option(CVPROJ_ENABLE_ARAVIS "Enable Aravis USB3 Vision / GenICam integration" ON)

if(NOT OpenCV_DIR AND EXISTS "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
    set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED COMPONENTS core dnn imgproc highgui videoio video)
find_package(PkgConfig REQUIRED)

add_subdirectory(cpp)
