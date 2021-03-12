set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set source files
set(SOURCE
        inclination.cpp
        maskgeneration.cpp
        reader.cpp
        toolbox.cpp
        writer.cpp
        version.cpp
        cuda/cuda_toolbox.cu)

# Set header files
set(HEADER
        inclination.h
        maskgeneration.h
        reader.h
        toolbox.h
        writer.h
        version.h
        cuda/cuda_toolbox.h
        cuda/define.h)

# Generate executable file
add_library(PLImig SHARED ${SOURCE})
target_link_libraries(PLImig ${OpenCV_LIBRARIES} CLI11::CLI11 ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} ${NIFTI_LIBRARIES}
        CUDA::cudart CUDA::nppidei CUDA::nppif CUDA::nppc CUDA::nppisu CUDA::nppist
        OpenMP::OpenMP_CXX OpenMP::OpenMP_C std::filesystem ${PLIM_LIBRARIES})