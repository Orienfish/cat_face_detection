#!/bin/sh
###################################################################
# generate tfRecord for tensorflow training
###################################################################
# Called by ./generate_tfrecord.sh in cat_dataset directory
# define path variables
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

###################################################################
# local training
###################################################################
# define path variables
PATH_TO_YOUR_PIPELINE_CONFIG=/home/robot/cat_dataset/training/ssd_mobilenet_v1_cat.config
PATH_TO_TRAIN_DIR=/home/robot/cat_dataset/training/trainlog/
PATH_TO_EVAL_DIR=/home/robot/cat_dataset/training/evallog/

# train
# ${PATH_TO_YOUR_PIPELINE_CONFIG} points to the pipeline config
# ${PATH_TO_TRAIN_DIR} points to the directory in which training checkpoints and events will be written to
python /usr/local/lib/python2.7/dist-packages/models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
echo "Finish local training!"
echo ""

###################################################################
# evaluate for only one time after training
###################################################################
# ${PATH_TO_EVAL_DIR} points to the directory in which evaluation events will be saved
python /usr/local/lib/python2.7/dist-packages/models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}
echo "Finish one-time evaluation!"
echo ""

###################################################################
# This script is for exporting the trained model
###################################################################
# define path variables
# note: the specific number is necessary!
PATH_TO_TRAINED_MODEL=/home/robot/cat_dataset/training/trainlog/model.ckpt-29932 
EXPORT_DIR=/home/robot/cat_dataset/training/exported_model/
# export
python /usr/local/lib/python2.7/dist-packages/models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --trained_checkpoint_prefix ${PATH_TO_TRAINED_MODEL} \
    --output_directory ${EXPORT_DIR}
echo "Finish exporting models!"
echo ""

###################################################################
# Testing
###################################################################
python my_object_detection.py /home/robot/cat_dataset/images /home/robot/cat_dataset/results
echo "Finish testing!"