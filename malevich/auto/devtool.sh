#!/bin/bash

# Check whether devtool is available

if ! [ -x "$(command -v jls-devtool)" ]; then
    echo "Devtool is not installed. Please install devtool and try again."
    exit 1
fi

# Check whether the script is being run in the correct directory
RUN_DIR=$(pwd)

# If does not contain the string "malevich/auto", but
# file malevich/auto/build.sh exists, then we are in the correct directory

if [[ $RUN_DIR == *"malevich/auto"* ]] && [! [ -f malevich/auto/build.sh ]]; then
    echo "Please run the script from the root See and Tell directory"
    exit 1
fi


bash malevich/auto/build.sh
jls-devtool

