#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

python -m alfred.json_generation \
--data_root="${PATH_DIR}"/data/alfred_splits/ \
--split='task_0-10'  \
--output_file="${PATH_DIR}"/data/object_detection_jsons/task_0-10.json