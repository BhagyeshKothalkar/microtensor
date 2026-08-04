cmake_minimum_required(VERSION 3.14)

project(microtensor VERSION 1.0 LANGUAGES CXX)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING
        "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel)"
        FORCE)
    message(STATUS "No build type selected, defaulting to Debug")
endif()

# Enforce strict C++26
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for language servers and IDE tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Apply strict warnings and turn them into errors in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-Wall -Wextra -Wpedantic)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/W4 /WX)
    endif()
endif()

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
option(ENABLE_TESTING "Enable unit testing" OFF)
option(ENABLE_DOCUMENTATION "Enable documentation" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ------------------------------------------------------------------
# Source
# ------------------------------------------------------------------
add_subdirectory(src)

# ------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------
if(ENABLE_TESTING)
    message(STATUS "Building tests...")

    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        5f9ad7d401161ee2249f31e857edb920a1185234
    )

    # Prevent overriding the parent project's compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_subdirectory(tests)
endif()

# ------------------------------------------------------------------
# Documentation
# ------------------------------------------------------------------
if(ENABLE_DOCUMENTATION)
    message(STATUS "Documentation build is not yet implemented.")
endif()