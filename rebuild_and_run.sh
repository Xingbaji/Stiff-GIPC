#!/bin/bash
set -e

# Rebuild
echo "Building..."
./build.sh

# Run
echo "Running simulation..."
# Default values can be overridden by environment variables or arguments
DEMO=${1:-1}
FRAMES=${2:-10}

echo "Using Demo: $DEMO, Frames: $FRAMES"
./build/gipc --headless --demo $DEMO --frames $FRAMES

