cmake_minimum_required(VERSION 3.20)
project(mnist_inference)

set(Torch_DIR "/home/holoscan/libtorch/share/cmake/Torch")

# Add NVTX subdirectory - adjust path based on where you cloned NVTX
add_subdirectory(NVTX)

# Find required packages
find_package(holoscan REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define source files
set(SOURCES
    inference.cpp
    mnist_app.cpp
)

# Create executable
add_executable(mnist_app ${SOURCES})

# Include directories
target_include_directories(mnist_app
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOLOSCAN_INCLUDE_DIRS}
    /opt/nvidia/holoscan/include/holoinfer
)

# Link libraries
target_link_libraries(mnist_app
    PRIVATE
    holoscan::core
    ${TORCH_LIBRARIES}
    TorchVision::TorchVision
    CUDA::cudart
    holoscan::ops::ping_tx  # If you're using ping operators
    holoscan::ops::ping_rx
    holoscan::infer
    holoscan::infer_utils
    nvtx3-cpp  # Link with NVTX C++ API
)