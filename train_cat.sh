#!/bin/sh
# This script is for local training

# define path variables
PATH_TO_YOUR_PIPELINE_CONFIG=/home/robot/cat_dataset/training/ssd_mobilenet_v1_cat.config
PATH_TO_TRAIN_DIR=/home/robot/cat_dataset/training/trainlog/
# PATH_TO_EVAL_DIR=/home/robot/cat_dataset/training/evallog/

# train
# From the tensorflow/models/research/ directory
# ${PATH_TO_YOUR_PIPELINE_CONFIG} points to the pipeline config
# ${PATH_TO_TRAIN_DIR} points to the directory in which training checkpoints and events will be written to
# ${PATH_TO_TRAIN_DIR} points to the directory in which training checkpoints and events will be written to
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}



