add_subdirectory(microtensor)

# 2. Define the main executable
add_executable(microtensor_app main.cpp)

# 3. Link the executable to the library
# This automatically inherits the include directories defined in the library!
target_link_libraries(microtensor_app PRIVATE microtensor_lib)
