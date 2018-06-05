#!/bin/sh
# This script is for exporting the trained model

# define path variables
PATH_TO_YOUR_PIPELINE_CONFIG=/home/robot/cat_dataset/training/ssd_mobilenet_v1_cat.config
# note: the specific number is necessary!
PATH_TO_TRAIN_DIR=/home/robot/cat_dataset/training/trainlog/model.ckpt-29932 
# PATH_TO_EVAL_DIR=/home/robot/cat_dataset/training/evallog/
EXPORT_DIR=/home/robot/cat_dataset/training/exported_model/
# export
# From tensorflow/models/research/
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --trained_checkpoint_prefix ${PATH_TO_TRAIN_DIR} \
    --output_directory ${EXPORT_DIR}