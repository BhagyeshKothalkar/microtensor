# 1. Change from INTERFACE to STATIC (or OBJECT)
add_library(microtensor_lib STATIC
    cpu_kernels.cpp     # Put your implementation file here
)

# Specify include directories for both building and consuming
target_include_directories(microtensor_lib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

# Add headers for IDE visibility (optional)
target_sources(microtensor_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tensor.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tensor_iterator.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/broadcasting.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu_kernels.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nn.hpp>
)

