set(CVPROJ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CVPROJ_SOURCES
    src/aravis_frame_source.cpp
    src/bytetrack_tracker.cpp
    src/main.cpp
    src/motion_pipeline.cpp
    src/opencv_video_source.cpp
    src/hikrobot_mvs_source.cpp
    src/mjpeg_server.cpp
    src/yolo_onnx_detector.cpp
    src/yolo_rknn_detector.cpp
    src/socketcan_gimbal.cpp
    src/target_state_filter.cpp
    src/realsense_t265_source.cpp
)

add_executable(cvproj_capture ${CVPROJ_SOURCES})

target_include_directories(cvproj_capture
    PRIVATE
        ${CVPROJ_INCLUDE_DIR}
)

target_link_libraries(cvproj_capture
    PRIVATE
        ${OpenCV_LIBS}
)

find_path(RKNN_INCLUDE_DIR
    NAMES rknn_api.h
    HINTS
        ${CMAKE_SOURCE_DIR}/third_party/rknn/include
        /usr/local/include
        /usr/include
)
find_library(RKNN_RUNTIME_LIBRARY
    NAMES rknnrt librknnrt
    HINTS
        ${CMAKE_SOURCE_DIR}/third_party/rknn/lib
        /usr/local/lib
        /usr/lib
        /usr/lib/aarch64-linux-gnu
)
if(RKNN_INCLUDE_DIR AND RKNN_RUNTIME_LIBRARY)
    target_compile_definitions(cvproj_capture PRIVATE CVPROJ_HAS_RKNN=1)
    target_include_directories(cvproj_capture PRIVATE ${RKNN_INCLUDE_DIR})
    target_link_libraries(cvproj_capture PRIVATE ${RKNN_RUNTIME_LIBRARY})
    get_filename_component(RKNN_RUNTIME_LIB_DIR "${RKNN_RUNTIME_LIBRARY}" DIRECTORY)
    set_target_properties(cvproj_capture PROPERTIES
        BUILD_RPATH "${RKNN_RUNTIME_LIB_DIR}"
        INSTALL_RPATH "${RKNN_RUNTIME_LIB_DIR}"
    )
    message(STATUS "RKNN runtime enabled")
    message(STATUS "  include: ${RKNN_INCLUDE_DIR}")
    message(STATUS "  library: ${RKNN_RUNTIME_LIBRARY}")
else()
    message(WARNING "RKNN runtime not found. Build will not support --detector rknn.")
endif()

if(CVPROJ_ENABLE_ARAVIS)
    pkg_check_modules(ARAVIS IMPORTED_TARGET aravis-0.8)
    if(ARAVIS_FOUND)
        target_compile_definitions(cvproj_capture PRIVATE CVPROJ_HAS_ARAVIS=1)
        target_link_libraries(cvproj_capture PRIVATE PkgConfig::ARAVIS)
        message(STATUS "Aravis enabled")
    else()
        message(WARNING "Aravis not found. Build will not support standard USB3 Vision / GenICam backend.")
    endif()
endif()

if(CVPROJ_ENABLE_REALSENSE)
    find_path(REALSENSE2_INCLUDE_DIR
        NAMES librealsense2/rs.hpp
        HINTS
            $ENV{HOME}/opt/librealsense/include
            /usr/local/include
            /usr/include
    )
    find_library(REALSENSE2_LIBRARY
        NAMES realsense2 librealsense2
        HINTS
            $ENV{HOME}/opt/librealsense/lib
            /usr/local/lib
            /usr/lib
            /usr/lib/aarch64-linux-gnu
    )
    if(REALSENSE2_INCLUDE_DIR AND REALSENSE2_LIBRARY)
        target_compile_definitions(cvproj_capture PRIVATE CVPROJ_HAS_REALSENSE2=1)
        target_include_directories(cvproj_capture PRIVATE ${REALSENSE2_INCLUDE_DIR})
        target_link_libraries(cvproj_capture PRIVATE ${REALSENSE2_LIBRARY})
        get_filename_component(REALSENSE2_LIB_DIR "${REALSENSE2_LIBRARY}" DIRECTORY)
        set_target_properties(cvproj_capture PROPERTIES
            BUILD_RPATH "${REALSENSE2_LIB_DIR}"
            INSTALL_RPATH "${REALSENSE2_LIB_DIR}"
        )
        message(STATUS "RealSense/librealsense2 enabled")
        message(STATUS "  include: ${REALSENSE2_INCLUDE_DIR}")
        message(STATUS "  library: ${REALSENSE2_LIBRARY}")
    else()
        pkg_check_modules(REALSENSE2 IMPORTED_TARGET realsense2)
        if(REALSENSE2_FOUND)
            target_compile_definitions(cvproj_capture PRIVATE CVPROJ_HAS_REALSENSE2=1)
            target_link_libraries(cvproj_capture PRIVATE PkgConfig::REALSENSE2)
            message(STATUS "RealSense/librealsense2 enabled")
        else()
            message(WARNING "librealsense2 not found. Build will not support --backend t265/realsense.")
        endif()
    endif()
endif()

if(CVPROJ_ENABLE_HIKROBOT_MVS)
    set(MVS_ROOT "$ENV{HIKROBOT_MVS_ROOT}" CACHE PATH "Hikrobot MVS SDK root")
    find_path(HIKROBOT_MVS_INCLUDE_DIR
        NAMES MvCameraControl.h
        HINTS
            ${MVS_ROOT}/include
            /opt/MVS/include
            /Applications/MVS.app/Contents/include
            /usr/local/include
    )
    find_library(HIKROBOT_MVS_LIBRARY
        NAMES MvCameraControl libMvCameraControl
        HINTS
            ${MVS_ROOT}/lib
            ${MVS_ROOT}/lib/64
            /opt/MVS/lib
            /usr/local/lib
            /Applications/MVS.app/Contents/lib
    )

    if(HIKROBOT_MVS_INCLUDE_DIR AND HIKROBOT_MVS_LIBRARY)
        set(HIKROBOT_MVS_ARCH_OK TRUE)
        if(APPLE)
            execute_process(
                COMMAND /usr/bin/lipo -archs "${HIKROBOT_MVS_LIBRARY}"
                OUTPUT_VARIABLE HIKROBOT_MVS_ARCHS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            set(CVPROJ_TARGET_ARCH "${CMAKE_HOST_SYSTEM_PROCESSOR}")
            if(CMAKE_OSX_ARCHITECTURES)
                list(GET CMAKE_OSX_ARCHITECTURES 0 CVPROJ_TARGET_ARCH)
            endif()
            if(HIKROBOT_MVS_ARCHS AND NOT HIKROBOT_MVS_ARCHS MATCHES "(^| )${CVPROJ_TARGET_ARCH}($| )")
                set(HIKROBOT_MVS_ARCH_OK FALSE)
                message(WARNING
                    "Hikrobot MVS SDK found, but ${HIKROBOT_MVS_LIBRARY} only provides [${HIKROBOT_MVS_ARCHS}] "
                    "while the current build targets [${CVPROJ_TARGET_ARCH}]. Direct Hikrobot linking is disabled for this build."
                )
            endif()
        endif()

        if(HIKROBOT_MVS_ARCH_OK)
            get_filename_component(HIKROBOT_MVS_LIB_DIR "${HIKROBOT_MVS_LIBRARY}" DIRECTORY)
            target_compile_definitions(cvproj_capture PRIVATE CVPROJ_HAS_HIKROBOT_MVS=1)
            target_include_directories(cvproj_capture PRIVATE ${HIKROBOT_MVS_INCLUDE_DIR})
            target_link_libraries(cvproj_capture PRIVATE ${HIKROBOT_MVS_LIBRARY})
            set_target_properties(cvproj_capture PROPERTIES
                BUILD_RPATH "${HIKROBOT_MVS_LIB_DIR}"
                INSTALL_RPATH "${HIKROBOT_MVS_LIB_DIR}"
            )
            message(STATUS "Hikrobot MVS SDK enabled")
            message(STATUS "  include: ${HIKROBOT_MVS_INCLUDE_DIR}")
            message(STATUS "  library: ${HIKROBOT_MVS_LIBRARY}")
        endif()
    else()
        message(WARNING "Hikrobot MVS SDK not found. Build will keep OpenCV/video fallback only.")
    endif()
endif()
