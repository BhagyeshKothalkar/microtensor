# INTERFACE means this is a header-only library. It produces no compiled binary.
add_library(microtensor_lib INTERFACE)

# Define where consumers of this library should look for headers.
# By pointing to the parent of the current directory (${CMAKE_CURRENT_SOURCE_DIR}/..),
# consumers can use: #include "microtensor/tensor.hpp"
target_include_directories(microtensor_lib INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

# Optional but highly recommended: add headers as sources so they show up in IDEs like VS Code / CLion
target_sources(microtensor_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tensor.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tensor_iterator.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/broadcasting.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu_kernels.hpp>
)

