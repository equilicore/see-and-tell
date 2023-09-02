#!/bin/bash

# Check whether the script is being run in the correct directory
RUN_DIR=$(pwd)

# If does not contain the string "malevich/auto", but
# file malevich/auto/build.sh exists, then we are in the correct directory

if [[ $RUN_DIR == *"malevich/auto"* ]] && [! [ -f malevich/auto/build.sh ]]; then
    echo "Please run the script from the root See and Tell directory"
    exit 1
fi

docker build -t see-and-tell-malevich -f malevich/Dockerfile .
