#!/bin/bash

if command -v python3 &>/dev/null; then
    echo "Python3 is already installed: $(python3 --version)"
else
    echo "Python3 is not installed. Installing now..."

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # For Debian-based distros (Ubuntu, Debian)
        if command -v apt-get &>/dev/null; then
            sudo apt-get update && sudo apt-get install -y python3
        # For RedHat-based distros (CentOS, Fedora)
        elif command -v yum &>/dev/null; then
            sudo yum install -y python3
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y python3
        fi

    else
      echo "Unsupported OS. Please install Python manually."
        exit 1
    fi
fi

PARENT_DIR="$(dirname "$PWD")"
echo $PARENT_DIR
cd $PARENT_DIR/pytorch

worldsize=$1
server_logdir=$2

python3 -m helper.dynamicbatching --world-size=$worldsize --server-logdir=$server_logdir