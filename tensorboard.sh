#!/bin/sh
# This script is for running tensorboard
# Can be run any where
PATH_TO_MODEL_DIRECTORY=/home/robot/cat_dataset/training/
HOST=162.105.93.130
PORT=6099

# tensorboard
# ${PATH_TO_MODEL_DIRECTORY} points to the directory that contains the train and eval directories
tensorboard --host=${HOST} --port=${PORT} --logdir=${PATH_TO_MODEL_DIRECTORY}