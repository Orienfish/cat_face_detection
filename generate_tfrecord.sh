#!/bin/sh
# This script is to generate tfRecord for tensorflow training
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
