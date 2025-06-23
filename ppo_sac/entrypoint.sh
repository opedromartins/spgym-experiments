#!/bin/bash

# Check if the package directory exists and contains a setup.py file
if [ -f "/sliding-puzzles-gym/setup.cfg" ]; then
    echo "Installing Sliding Puzzles..."
    pip3 install -e /sliding-puzzles-gym
    sliding-puzzles setup diffusiondb
    sliding-puzzles setup imagenet
else
    echo "Sliding Puzzles not found. Mount it as a volume at /sliding-puzzles-gym"
fi

# Execute the original command
exec "$@"