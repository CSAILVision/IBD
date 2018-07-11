#!/usr/bin/env bash
set -e

# Start from parent directory of script

echo "Downloading"
mkdir -p zoo
pushd zoo
wget http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar
popd

echo "done"
