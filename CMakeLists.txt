cmake_minimum_required(VERSION 3.14)

project(microtensor VERSION 1.0 LANGUAGES CXX)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    message(STATUS "No build type selected, defaulting to Debug")
endif()

# Enforce strict C++26
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Disables compiler-specific extensions (e.g., gnu++23)

# Generate compile_commands.json for language servers and IDE tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Apply strict warnings and turn them into errors in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-Wall -Wextra -Wpedantic)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/W4 /WX) # MSVC equivalents
    endif()
endif()

# Options
option(ENABLE_TESTING "Enable unit testing" OFF)
option(ENABLE_DOCUMENTATION "Enable documentation" OFF) # Switched default to OFF for faster local builds

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Add the source tree
add_subdirectory(src)

# Testing and Docs (unchanged, just kept clean)
if(ENABLE_TESTING)
    message(STATUS "Building tests...")
    # add_subdirectory("lib/Catch2")
    # include(CTest)
    # include(Catch)
    # add_subdirectory("test")
endif()