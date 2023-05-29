#!/usr/bin/env bash

podman run \
  -v $GIBSON_ASSETS_PATH:$GIBSON_ASSETS_PATH \
  -v $IGIBSON_DATASET_PATH:$IGIBSON_DATASET_PATH \
  --env "GIBSON_ASSETS_PATH" \
  --env "IGIBSON_DATASET_PATH" \
  -it ssg
