if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

cmake_minimum_required(VERSION 3.14)
project(PLImg LANGUAGES C CXX CUDA)

# Set a default build type if none was specified
set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80)
endif()

# Search for required packages and load them
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
find_package(CUDAToolkit REQUIRED)
find_package(TIFF REQUIRED)
find_package(NIFTI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Filesystem REQUIRED)

# Add headers to project
add_subdirectory(extern/CLI11)
include_directories(extern/CLI11/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${NIFTI_INCLUDE_DIRS})
include_directories(${HDF5_INCLUDE_DIR})
include_directories(${OPENMP_C_INCLUDE_DIRS})
include_directories(${OPENMP_CXX_INCLUDE_DIRS})
include_directories(src/)

# Set CXX and C flags for OpenMP
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS})

add_subdirectory(extern)
add_subdirectory(src)
add_subdirectory(bin)
add_subdirectory(docs)

if(BUILD_TESTING)
   enable_testing()
   add_subdirectory(tests)
endif()
