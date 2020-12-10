find_package(GTest)

# Search for required packages and load them
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(src/)

add_executable(test_reader test_reader.cpp ${PROJECT_SOURCE_DIR}/src/reader.cpp)
target_link_libraries(test_reader GTest::GTest ${OpenCV_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} ${NIFTI_LIBRARIES})
gtest_discover_tests(test_reader TEST_PREFIX new:)

add_executable(test_writer test_writer.cpp ${PROJECT_SOURCE_DIR}/src/writer.cpp ${PROJECT_SOURCE_DIR}/src/reader.cpp)
target_link_libraries(test_writer GTest::GTest ${OpenCV_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} ${NIFTI_LIBRARIES})
gtest_discover_tests(test_writer TEST_PREFIX new:)

add_executable(test_toolbox test_toolbox.cpp ${PROJECT_SOURCE_DIR}/src/toolbox.cpp ${PROJECT_SOURCE_DIR}/src/cuda/toolbox.cu)
target_link_libraries(test_toolbox GTest::GTest ${OpenCV_LIBRARIES} CUDA::cudart CUDA::nppidei CUDA::nppif CUDA::nppc CUDA::nppisu)
gtest_discover_tests(test_toolbox TEST_PREFIX new:)

add_executable(test_maskgeneration test_maskgeneration.cpp ${PROJECT_SOURCE_DIR}/src/toolbox.cpp ${PROJECT_SOURCE_DIR}/src/maskgeneration.cpp ${PROJECT_SOURCE_DIR}/src/cuda/toolbox.cu)
target_link_libraries(test_maskgeneration GTest::GTest ${OpenCV_LIBRARIES} CUDA::cudart CUDA::nppidei CUDA::nppif CUDA::nppc CUDA::nppisu OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
gtest_discover_tests(test_maskgeneration TEST_PREFIX new:)

if(CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(test_reader gcov)
    target_link_libraries(test_writer gcov)
    target_link_libraries(test_toolbox gcov)
    target_link_libraries(test_maskgeneration gcov)

    include(CodeCoverage)
    set(COVERAGE_EXCLUDES "extern/*/*/*" "extern/*/*")
    APPEND_COVERAGE_COMPILER_FLAGS()
    setup_target_for_coverage_gcovr_html(NAME coverage EXECUTABLE ctest)
endif()
