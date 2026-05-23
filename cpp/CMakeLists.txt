cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CENTRAL_CONTROL_ENABLE_ORT "Enable ONNX Runtime integration" ON)
option(CENTRAL_CONTROL_ENABLE_MVS "Enable Hikrobot MVS SDK integration" ON)
option(CENTRAL_CONTROL_ENABLE_PCAN "Enable PEAK PCAN integration" ON)
option(CENTRAL_CONTROL_ENABLE_RKNN "Enable RKNN integration placeholders" OFF)

set(CENTRAL_CONTROL_MVS_ROOT "C:/Program Files (x86)/MVS/Development" CACHE PATH "Hikrobot MVS Development root")
set(CENTRAL_CONTROL_ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime root")
set(CENTRAL_CONTROL_PCAN_ROOT "" CACHE PATH "PEAK PCAN root")

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio video dnn)

set(CENTRAL_CONTROL_SOURCES
    src/common.cpp
    src/components.cpp
    src/runtime.cpp
    src/main.cpp
)

add_executable(central_control ${CENTRAL_CONTROL_SOURCES})
target_include_directories(central_control
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(central_control PRIVATE ${OpenCV_LIBS})

if(WIN32)
    target_compile_definitions(central_control PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

if(CENTRAL_CONTROL_ENABLE_MVS AND EXISTS "${CENTRAL_CONTROL_MVS_ROOT}/Includes/MvCameraControl.h")
    target_include_directories(central_control PRIVATE "${CENTRAL_CONTROL_MVS_ROOT}/Includes")
    if(WIN32 AND EXISTS "${CENTRAL_CONTROL_MVS_ROOT}/Libraries/win64/MvCameraControl.lib")
        target_link_libraries(central_control PRIVATE "${CENTRAL_CONTROL_MVS_ROOT}/Libraries/win64/MvCameraControl.lib")
    endif()
    target_compile_definitions(central_control PRIVATE CENTRAL_CONTROL_WITH_MVS=1)
endif()

if(CENTRAL_CONTROL_ENABLE_ORT AND EXISTS "${CENTRAL_CONTROL_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    target_include_directories(central_control PRIVATE "${CENTRAL_CONTROL_ONNXRUNTIME_ROOT}/include")
    if(WIN32 AND EXISTS "${CENTRAL_CONTROL_ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
        target_link_libraries(central_control PRIVATE "${CENTRAL_CONTROL_ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    elseif(EXISTS "${CENTRAL_CONTROL_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
        target_link_libraries(central_control PRIVATE "${CENTRAL_CONTROL_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    endif()
    target_compile_definitions(central_control PRIVATE CENTRAL_CONTROL_WITH_ORT=1)
endif()

if(CENTRAL_CONTROL_ENABLE_PCAN AND CENTRAL_CONTROL_PCAN_ROOT AND EXISTS "${CENTRAL_CONTROL_PCAN_ROOT}")
    target_include_directories(central_control PRIVATE "${CENTRAL_CONTROL_PCAN_ROOT}/include")
    if(WIN32 AND EXISTS "${CENTRAL_CONTROL_PCAN_ROOT}/x64/VC_LIB/PCANBasic.lib")
        target_link_libraries(central_control PRIVATE "${CENTRAL_CONTROL_PCAN_ROOT}/x64/VC_LIB/PCANBasic.lib")
    endif()
    target_compile_definitions(central_control PRIVATE CENTRAL_CONTROL_WITH_PCAN=1)
endif()

if(CENTRAL_CONTROL_ENABLE_RKNN)
    target_compile_definitions(central_control PRIVATE CENTRAL_CONTROL_WITH_RKNN=1)
endif()
