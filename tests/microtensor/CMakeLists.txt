include(GoogleTest)

add_executable(microtensor_tests 
    tensors.cpp        
    tensor_iterator.cpp
    broadcasting.cpp
    functional.cpp
    nn.cpp
    cpu_kernels.cpp    
    autograd.cpp
)

target_include_directories(microtensor_tests PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(microtensor_tests PRIVATE 
    microtensor_lib    # library
    GTest::gtest_main  # GoogleTest + standard main loop
)

gtest_discover_tests(microtensor_tests)