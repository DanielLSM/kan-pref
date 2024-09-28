#!/bin/bash

# Set the PATH environment variable
export PATH="/opt/conda/envs/base/bin:$PATH"

# Activate Micromamba environment
eval "$(micromamba shell hook --shell=bash)"
micromamba activate

# Execute the provided command
exec "$@"