#!/bin/sh
set -e # exit if any return with non-zero
###################################################################
# This script is for exporting the trained model
###################################################################
# define path variables
# note: the specific number is necessary!
PATH_TO_YOUR_PIPELINE_CONFIG=/home/robot/cat_dataset/training/ssd_mobilenet_v1_cat.config
PATH_TO_TRAINED_MODEL=/home/robot/cat_dataset/training/trainlog/model.ckpt-20883 
EXPORT_DIR=/home/robot/cat_dataset/training/exported_model
# if export_dir exists, delete it
if [ -d "$EXPORT_DIR" ]; then
    rm -rf "$EXPORT_DIR"
fi
mkdir "$EXPORT_DIR" # new export_dir
# export
python /usr/local/lib/python2.7/dist-packages/tensorflow/models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --trained_checkpoint_prefix ${PATH_TO_TRAINED_MODEL} \
    --output_directory ${EXPORT_DIR}
echo "Finish exporting models!"

###################################################################
# Testing
###################################################################
# total_test_cnt should be less equal than 2600
python my_object_detection.py /home/robot/cat_dataset/images /home/robot/cat_dataset/results gpu
echo "Finish testing!"


