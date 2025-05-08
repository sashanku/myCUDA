#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake
cmake ..

# Build the project
make -j$(nproc)

# Optional: Run the executable
# ./benchmark

# Go back to the original directory
cd ..

echo "Build complete. The executable is located at ./build/benchmark"