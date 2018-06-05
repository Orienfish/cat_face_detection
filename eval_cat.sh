#!/bin/sh
# This script is for local evaluating

# define path variables
PATH_TO_YOUR_PIPELINE_CONFIG=/home/robot/cat_dataset/training/ssd_mobilenet_v1_cat.config
PATH_TO_TRAIN_DIR=/home/robot/cat_dataset/training/trainlog/
PATH_TO_EVAL_DIR=/home/robot/cat_dataset/training/evallog/

# evaluate
# From the tensorflow/models/research/ directory
# ${PATH_TO_YOUR_PIPELINE_CONFIG} points to the pipeline config
# ${PATH_TO_TRAIN_DIR} points to the directory in which training checkpoints were saved (same as the training job)
# ${PATH_TO_EVAL_DIR} points to the directory in which evaluation events will be saved
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}