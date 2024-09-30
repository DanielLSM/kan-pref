#!/bin/bash
export USE_GPU=False  # or False if you want to use CPU
export RELEASE=True  # or False if you don't want to push the image
PARENT=stablebaselines/stable-baselines3
VERSION=2.2.0a1

TAG=dlsm666/kan-pref
OURVERSION=2.5.0

if [[ ${USE_GPU} == "True" ]]; then
  PARENT="${PARENT}:${VERSION}"
else
  PARENT="${PARENT}-cpu:${VERSION}"
  TAG="${TAG}-cpu"
fi

docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg USE_GPU=${USE_GPU} -t ${TAG}:${OURVERSION} . -f docker/Dockerfile
docker tag ${TAG}:${OURVERSION} ${TAG}:latest

if [[ ${RELEASE} == "True" ]]; then
  docker push ${TAG}:${OURVERSION}
  docker push ${TAG}:latest
fi
sleep 30