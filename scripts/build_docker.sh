#!/bin/bash
export USE_GPU=False  # or False if you want to use CPU
export RELEASE=False  # or False if you don't want to push the image
PARENT=stablebaselines/stable-baselines3

TAG=sholk/rl-baselines3-new
VERSION=2.2.0a1

if [[ ${USE_GPU} == "True" ]]; then
  PARENT="${PARENT}:${VERSION}"
else
  PARENT="${PARENT}-cpu:${VERSION}"
  TAG="${TAG}-cpu"
fi

docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg USE_GPU=${USE_GPU} -t ${TAG}:${VERSION} . -f docker/Dockerfile
docker tag ${TAG}:${VERSION} ${TAG}:latest

if [[ ${RELEASE} == "True" ]]; then
  docker push ${TAG}:${VERSION}
  docker push ${TAG}:latest
fi
sleep 30