# Set output folders to bin and lib
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tools)

# other tools like benchmarks
add_executable(BlurredParameterTesting BlurredParameterTesting.cpp)
target_link_libraries(BlurredParameterTesting PLImig)

# other tools like benchmarks
add_executable(RegionGrowingTesting RegionGrowingTesting.cpp)
target_link_libraries(RegionGrowingTesting PLImig)

# other tools like benchmarks
add_executable(testConnectedComponentsImplementation testConnectedComponentsImplementation.cpp)
target_link_libraries(testConnectedComponentsImplementation PLImig)