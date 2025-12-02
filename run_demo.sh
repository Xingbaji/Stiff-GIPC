#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================"
echo "Step 1: Configuring CMake..."
echo "========================================"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "========================================"
echo "Step 2: Building Project..."
echo "========================================"
# Use all available cores for building
make -j$(nproc)

echo "========================================"
echo "Step 3: Running Demo..."
echo "========================================"
./gipc

echo "Done."
