#!/bin/bash

running_experiment_name=$1
args_experiment="${@:2}"

docker restart data-augmentation-review-container
docker exec data-augmentation-review-container bash -c "python3 -u main.py $args_experiment > $running_experiment_name".out""
