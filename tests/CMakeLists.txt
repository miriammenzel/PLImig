find_package(GTest)

#add_executable(test_hello_world test_hello_world.cpp)
#target_link_libraries(test_hello_world GTest::GTest)
#gtest_discover_tests(test_hello_world TEST_PREFIX new:)

if(CMAKE_COMPILER_IS_GNUCXX)
    #target_link_libraries(test_hello_world gcov )
    include(CodeCoverage)
    set(COVERAGE_EXCLUDES "extern/*/*/*" "extern/*/*")
    APPEND_COVERAGE_COMPILER_FLAGS()
    setup_target_for_coverage_gcovr_html(NAME coverage EXECUTABLE ctest)
endif()
