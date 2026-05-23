cmake_minimum_required(VERSION 3.21)

project(CVProj2026 LANGUAGES CXX)

option(CENTRAL_CONTROL_BUILD_CPP "Build the new C++ central-control runtime" ON)

if(CENTRAL_CONTROL_BUILD_CPP)
    add_subdirectory(cpp)
endif()
