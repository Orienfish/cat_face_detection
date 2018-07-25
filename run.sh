#!/bin/sh
set -e # exit if any return with non-zero
export PATH=$PATH:/home/robot/cocoapi/PythonAPI # add cocoapi to path, run this in terminal if coco import error

###################################################################
# generate tfRecord for tensorflow training
###################################################################
# Called by ./generate_tfrecord.sh in cat_dataset directory
# define path variables
# NO SPACE NEXT TO '='!!!!!!!
TRAIN_XML_PATH=annotations/train
TEST_XML_PATH=annotations/test
TRAIN_CSV_PATH=data/train_labels.csv
TEST_CSV_PATH=data/test_labels.csv
TRAIN_TFRECORD_PATH=data/train.record
TEST_TFRECORD_PATH=data/test.record
# generate xml to csv
python xml_to_csv.py --trainxml=${TRAIN_XML_PATH} --testxml=${TEST_XML_PATH} \
	--traincsv=${TRAIN_CSV_PATH} --testcsv=${TEST_CSV_PATH}
# for generate tfRecord, call generate_tfrecord.py
python generate_tfrecord.py --csv_input=${TRAIN_CSV_PATH}  --output_path=${TRAIN_TFRECORD_PATH}
python generate_tfrecord.py --csv_input=${TEST_CSV_PATH}  --output_path=${TEST_TFRECORD_PATH}
echo "Finish tfRecord generation!"

# concurrently run train, test and tensorboard
###################################################################
# local training
###################################################################
# define path variables
PATH_TO_YOUR_PIPELINE_CONFIG=training/ssd_mobilenet_v1_cat.config
PATH_TO_TRAIN_DIR=training/trainlog
PATH_TO_EVAL_DIR=training/evallog

# remove previous record
if [ -d "$PATH_TO_TRAIN_DIR" ]; then
	rm -rf "$PATH_TO_TRAIN_DIR"
fi
mkdir "$PATH_TO_TRAIN_DIR" # new a folder
if [ -d "$PATH_TO_EVAL_DIR" ]; then
	rm -rf "$PATH_TO_EVAL_DIR"
fi
mkdir "$PATH_TO_EVAL_DIR" # new a folder
# kill previous processes
ps -ef | grep tensorflow | grep object_detection | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep tensorboard | grep 6066 | grep -v grep | awk '{print $2}' | xargs kill -9

# train
# ${PATH_TO_YOUR_PIPELINE_CONFIG} points to the pipeline config
# ${PATH_TO_TRAIN_DIR} points to the directory in which training checkpoints and events will be written to
# run this process in the background. use gpu:0. re-direct stdout and stderr to training/runlog/train.log
CUDA_VISIBLE_DEVICES=1 python /usr/local/lib/python2.7/dist-packages/tensorflow/models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR} >training/runlog/train.log 2>&1 &
echo "start training in the back!"

###################################################################
# evaluate
###################################################################
# ${PATH_TO_EVAL_DIR} points to the directory in which evaluation events will be saved
# run this process in the background. use gpu:1. re-direct stdout and stderr to training/runlog/eval.log
CUDA_VISIBLE_DEVICES=0 python /usr/local/lib/python2.7/dist-packages/tensorflow/models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR} >training/runlog/eval.log 2>&1 &
echo "start evaluating in the back!"

###################################################################
# running tensorboard
###################################################################
# run here in cat_dataset_new root directory
PATH_TO_MODEL_DIRECTORY=training/
HOST=162.105.93.130
PORT=6066

# tensorboard
# ${PATH_TO_MODEL_DIRECTORY} points to the directory that contains the train and eval directories
# run this process in the background. re-direct stdout and stderr to training/runlog/tensorboard.log
tensorboard --host=${HOST} --port=${PORT} --logdir=${PATH_TO_MODEL_DIRECTORY} >training/runlog/tensorboard.log 2>&1 &
echo "tensorboard"

###################################################################
# search for the pid of train and eval process
###################################################################
# ps -ef | grep 'tensorflow' | grep 'train.py' | grep -v 'grep'
pid_train=$(ps -ef | grep 'tensorflow' | grep 'train.py' | grep -v 'grep' | awk '{print $2}')
# ps -ef | grep 'tensorflow' | grep 'eval.py' | grep -v 'grep'
pid_eval=$(ps -ef | grep 'tensorflow' | grep 'eval.py' | grep -v 'grep' | awk '{print $2}')
echo "training pid is $pid_train, evaluating pid is $pid_eval"
# wait for train process to terminate
wait $pid_train 
echo "training process $child exited!"
# kill the eval process if necessary
ps -ef | grep 'tensorflow' | grep 'eval.py' | grep -v 'grep' | awk '{print $2}' | xargs kill -9

