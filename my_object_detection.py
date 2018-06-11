# coding: utf-8
"""
This program is for cat face detection, modified from object_detection_tutorial.py
Should be run from tensorflow/models/research/object_detection directory.
Users are responsible for specifying PATH_TO_CKPT, PATH_TO_LABELS and NUM_CLASSES
according to their own needs.
"""
import numpy as np
import os
# not use GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO

# This is needed since the program is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# Env setup
from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation 
# ## Variables
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/robot/cat_dataset/training/exported_model/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/robot/cat_dataset/training/label_map.pbtxt'
# num of classes: cat_face and cat_eyes
NUM_CLASSES = 2

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `1`, we know that this corresponds to `cat_face'.
# Here we use the prepared label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# # Detection Files Paths
# the path to test images and where to save the results
"""
PATH_TO_DIR = '/home/robot/cat_dataset'
PATH_TO_TEST_IMAGES_DIR = os.path.join(PATH_TO_DIR, "images")
PATH_TO_SAVE_IMAGES_DIR = os.path.join(PATH_TO_DIR, "results")
if not os.path.isdir(PATH_TO_SAVE_IMAGES_DIR):
  os.makedirs(PATH_TO_SAVE_IMAGES_DIR)
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'cat.{}.jpg'.format(i)) for i in range(1, 100) ]
TEST_IMAGE_PATHS = ["/home/robot/CatCamera_Master/server/upload_files/3.jpg"]
"""
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

"""
run_inference_for_single_image - detect one frame
"""
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

'''
one_face_detection
Input: image - the object read from imread
Return: [xmin, xmax, ymin, ymax] 
'''
def one_face_detection(image):
    # check whether image is empty
    if (image is None):
    	print("The image is None!")
    	return 1.0, 1.0, 1.0, 1.0
    # Actual detection.
    output_dict = run_inference_for_single_image(image, detection_graph)
    # find the first cat face, i is the result index
    i = 0
    classes = output_dict['detection_classes'][0] 
    while classes != 1:
    	i += 1
    	classes = output_dict['detection_classes'][i]
    score = output_dict['detection_scores'][i]
    # check the detection score
    if score < 0.01:
    	return 1.0, 1.0, 1.0, 1.0
    # successfully detect a cat face
    return output_dict['detection_boxes'][i]

'''
multiple_face_detection
Input: image - the object read from imread
Return: [[xmin, xmax, ymin, ymax], [xmin, xmax, ymin, ymax], ...]
'''
def multiple_face_detection(image):
    # check whether image is empty
    if (image is None):
    	print("The image is None!")
    	return 1.0, 1.0, 1.0, 1.0
    # Actual detection.
    output_dict = run_inference_for_single_image(image, detection_graph)
    # find all cat faces, i is the index
    i = 0
    index = []
    classes = output_dict['detection_classes'][0]
    scores = output_dict['detection_scores'][0]
    while scores > 0.01:
    	if classes == 1: # detect cat face! 
        	index.append(i)
        # update
    	i += 1
    	classes = output_dict['detection_classes'][i]
    	scores = output_dict['detection_scores'][i]
    # no cat face detect!
    if index is None:
        return 1.0, 1.0, 1.0, 1.0
    # successfully detect cat faces
    res = []
    for j in range(len(index)):
	res.append(output_dict['detection_boxes'][j])
    return res

"""
main routine
"""
def main():
  cnt = 0
  for image_path in TEST_IMAGE_PATHS:
    print image_path
    # load image
    image_np = cv2.imread(image_path)
    print(one_face_detection(image_np))
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    print("processing %s" %image_path)
    # print(output_dict['detection_boxes'])
    # print(output_dict['detection_classes'])
    # print(output_dict['detection_scores'])
    # write back processed images
    # cv2.imwrite(os.path.join(PATH_TO_SAVE_IMAGES_DIR, 'cat.{}.jpg'.format(cnt)), image_np)
    cv2.imshow("img", image_np)
    cv2.waitKey(0)
    cnt += 1

if __name__ == '__main__':
  main()
