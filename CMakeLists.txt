if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

cmake_minimum_required(VERSION 3.14)
project(PLImg LANGUAGES C CXX CUDA)

# Set a default build type if none was specified
set(default_build_type "Release")

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    set(default_build_type "Debug")

    # Update submodules as needed
    option(PLIMG_SUBMODULE "Check submodules during build" ON)
    if(PLIMG_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(
                    FATAL_ERROR
                    "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules"
            )
        endif()
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Search for required packages and load them
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(CUDAToolkit REQUIRED)
find_package(TIFF REQUIRED)
find_package(NIFTI REQUIRED)

# Add headers to project
add_subdirectory(extern/CLI11)
include_directories(extern/CLI11/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${NIFTI_INCLUDE_DIRS})
include_directories(src/)

# Generate executable file
add_executable(PLImg main.cpp src/reader.cpp src/writer.cpp src/maskgeneration.cpp src/toolbox.cpp)
target_link_libraries(PLImg ${OpenCV_LIBRARIES} CLI11::CLI11 ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} ${NIFTI_LIBRARIES}
                            CUDA::cudart CUDA::nppidei CUDA::nppif CUDA::nppc CUDA::nppisu)

if(BUILD_TESTING)
   enable_testing()
   add_subdirectory(tests)
endif()