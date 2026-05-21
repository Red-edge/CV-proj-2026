set(CVPROJ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CVPROJ_SOURCES
    src/aravis_frame_source.cpp
    src/main.cpp
    src/motion_pipeline.cpp
    src/opencv_video_source.cpp
    src/hikrobot_mvs_source.cpp
    src/yolo_onnx_detector.cpp
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
