#!/bin/bash
cd ..
DOCKER_BUILDKIT=1 docker build --rm=true --target runtime --build-arg DEVICE=cuda --build-arg FREESURFER=pruned -t fastsurfer:gpu -f ./Docker/Dockerfile .
