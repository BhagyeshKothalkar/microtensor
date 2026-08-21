# Change from INTERFACE to STATIC (or OBJECT)
add_library(microtensor_lib STATIC
    autograd.cpp
    cpu.cpp
    functional.cpp
    nn.cpp
    optim.cpp
    shape.cpp
    tensor.cpp
)

# Specify include directories for both building and consuming
target_include_directories(microtensor_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

# Add headers for IDE visibility (optional)
target_sources(microtensor_lib PUBLIC
    autograd.hpp
    cpu.hpp
    functional.hpp
    nn.hpp
    optim.hpp
    shape.hpp
    tensor.hpp
)
