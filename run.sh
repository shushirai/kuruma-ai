#!/bin/bash

docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  --name kuruma-ai-container \
  kuruma-ai:base
