add_executable(regtesting regtest.cpp)
add_executable(testing test.cpp)

target_link_libraries(testing PUBLIC microtensor_lib)
target_link_libraries(regtesting PUBLIC microtensor_lib)
