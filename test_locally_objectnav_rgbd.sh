#!/usr/bin/env bash

DOCKER_NAME="objectnav_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

dataset=$2
challenge=$3

# echo $dataset
# echo $challenge

docker run -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v /coc/dataset/habitat-sim-datasets/hm3d/:/habitat-challenge-data/data/scene_datasets/hm3d/ \
    -v $(pwd)/logs:/logs \
    --runtime=nvidia \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml" \
    ${DOCKER_NAME}\

