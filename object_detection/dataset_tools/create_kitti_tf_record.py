# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw KITTI detection dataset to TFRecord for object_detection.
Converts KITTI detection dataset to TFRecords with a standard format allowing
  to use this dataset to train object detectors. The raw dataset can be
  downloaded from:
  http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip.
  http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip
  Permission can be requested at the main website.
  KITTI detection dataset contains 7481 training images. Using this code with
  the default settings will set aside the first 500 images as a validation set.
  This can be altered using the flags, see details below.
Example usage:
    python object_detection/dataset_tools/create_kitti_tf_record.py \
        --data_dir=/home/user/kitti \
        --output_path=/home/user/kitti.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import hashlib
import io
import os

import numpy as np
import PIL.Image as pil
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils.np_box_ops import iou

tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                           'data. Folder structure is assumed to be:'
                           '<data_dir>/training/label_2 (annotations) and'
                           '<data_dir>/data_object_image_2/training/image_2'
                           '(images).')
tf.app.flags.DEFINE_string('output_path', '', 'Path to which TFRecord files'
                           'will be written. The TFRecord with the training set'
                           'will be located at: <output_path>_train.tfrecord.'
                           'And the TFRecord with the validation set will be'
                           'located at: <output_path>_val.tfrecord')
tf.app.flags.DEFINE_string('classes_to_use', 'car,pedestrian,cyclist,person_sitting,truck,misc,van,tram,dontcare',
                           'Comma separated list of class names that will be'
                           'used. Adding the dontcare class will remove all'
                           'bboxs in the dontcare regions.')
tf.app.flags.DEFINE_string('label_map_path', 'data/kitti_label_map.pbtxt',
                           'Path to label map proto.')
tf.app.flags.DEFINE_integer('validation_set_size', '481', 'Number of images to'
                            'be used as a validation set.')
tf.app.flags.DEFINE_integer('train_set_size', '300', 'Number of images to'
                            'be used as a validation set.')
FLAGS = tf.app.flags.FLAGS


def convert_kitti_to_tfrecords(data_dir, output_path, classes_to_use,
                               label_map_path, validation_set_size, train_set_size):
  """Convert the KITTI detection dataset to TFRecords.
  Args:
    data_dir: The full path to the unzipped folder containing the unzipped data
      from data_object_image_2 and data_object_label_2.zip.
      Folder structure is assumed to be: data_dir/training/label_2 (annotations)
      and data_dir/data_object_image_2/training/image_2 (images).
    output_path: The path to which TFRecord files will be written. The TFRecord
      with the training set will be located at: <output_path>_train.tfrecord
      And the TFRecord with the validation set will be located at:
      <output_path>_val.tfrecord
    classes_to_use: List of strings naming the classes for which data should be
      converted. Use the same names as presented in the KIITI README file.
      Adding dontcare class will remove all other bounding boxes that overlap
      with areas marked as dontcare regions.
    label_map_path: Path to label map proto
    validation_set_size: How many images should be left as the validation set.
      (Ffirst `validation_set_size` examples are selected to be in the
      validation set).
  """
  label_map_dict = label_map_util.get_label_map_dict(label_map_path)
  train_count = 0
  val_count = 0

  annotation_dir = os.path.join(data_dir,
                                'raw_data',
                                'training',
                                'label_2')

  image_dir = os.path.join(data_dir,
                           'raw_data',
                           'data_object_image_2',
                           'training',
                           'image_2')
  # FOR CREATING ALL TFRECORDS FOR ENTIRE SET TRAINING
  """  output_path_train = output_path + "/entire_set/"
  train_writer = tf.python_io.TFRecordWriter('%s_0_entire.tfrecord'%
                                             output_path_train)

  images_list = []
  with open(output_path+"/all_images.txt", "r") as f:
    images_list = f.readlines()

  tfrecord_count = 0
  for img_index, img_name in enumerate(images_list):
    img_num = int(img_name.split('.')[0])
    #is_validation_img = img_num < validation_set_size
    #print(validation_set_size)
    img_anno = read_annotation_file(os.path.join(annotation_dir,
                                                 str(img_num).zfill(6)+'.txt'))

    image_path = os.path.join(image_dir, img_name[:-1])
    
    # Filter all bounding boxes of this frame that are of a legal class, and
    # don't overlap with a dontcare region.
    # TODO(talremez) filter out targets that are truncated or heavily occluded.
    annotation_for_image = filter_annotations(img_anno, classes_to_use)

    example = prepare_example(image_path, annotation_for_image, label_map_dict)

     
    train_writer.write(example.SerializeToString())
    train_count += 1

    if train_count == 500:
      print("train_count ", train_count)
      train_writer.close()
      tfrecord_count += 1
      train_count = 0
      train_writer = tf.python_io.TFRecordWriter('%s_%s_entire.tfrecord'%
                                             (output_path_train,tfrecord_count))


  exit()"""
  output_path_train = output_path + "/train_set/"
  train_writer = tf.python_io.TFRecordWriter('%strain.tfrecord'%
                                             output_path_train)
  output_path_val = output_path + "/val_set/"
  val_writer = tf.python_io.TFRecordWriter('%sval.tfrecord'%
                                           output_path_val)

  #images = sorted(tf.gfile.ListDirectory(image_dir))

  images = tf.gfile.ListDirectory(image_dir)

  images_shuffled = np.random.permutation(images)
 
  split_from = int(train_set_size) + validation_set_size
  images_pretrain = images_shuffled[:split_from]
  #print(len(images_pretrain))
  images_al = images_shuffled[split_from:]

  with open('/home/aev21/data/KITTI/images_for_al2.txt', 'w') as f:
      for img in images_al:
          f.write("%s\n" % img[:10])
  f.close()

  with open('/home/aev21/data/KITTI/images_for_al_backup2.txt', 'w') as f:
    for img in images_al:
        f.write("%s\n" % img[:10])
  f.close()

  for img_index, img_name in enumerate(images_pretrain):
    img_num = int(img_name.split('.')[0])
    #is_validation_img = img_num < validation_set_size
    #print(validation_set_size)
    img_anno = read_annotation_file(os.path.join(annotation_dir,
                                                 str(img_num).zfill(6)+'.txt'))

    image_path = os.path.join(image_dir, img_name)
    
    # Filter all bounding boxes of this frame that are of a legal class, and
    # don't overlap with a dontcare region.
    # TODO(talremez) filter out targets that are truncated or heavily occluded.
    annotation_for_image = filter_annotations(img_anno, classes_to_use)

    example = prepare_example(image_path, annotation_for_image, label_map_dict)

    if img_index < validation_set_size:
      val_writer.write(example.SerializeToString())
      with open('/home/aev21/data/KITTI/images_for_val2.txt', 'a+') as f:
          f.write("%s\n" % img_name)
      f.close()
      val_count += 1

    else:
      train_writer.write(example.SerializeToString())
      with open('/home/aev21/data/KITTI/images_for_train2.txt', 'a+') as f:
          f.write("%s\n" % img_name)
      f.close()
      train_count += 1
   
  print("train_count ", train_count, "val_count ", val_count)
  train_writer.close()
  val_writer.close()
  
def prepare_example(image_path, annotations, label_map_dict):
  """Converts a dictionary with annotations for an image to tf.Example proto.
  Args:
    image_path: The complete path to image.
    annotations: A dictionary representing the annotation of a single object
      that appears in the image.
    label_map_dict: A map from string label names to integer ids.
  Returns:
    example: The converted tf.Example.
  """
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = pil.open(encoded_png_io)
  image = np.asarray(image)

  key = hashlib.sha256(encoded_png).hexdigest()

  width = int(image.shape[1])
  height = int(image.shape[0])

  xmin_norm = annotations['2d_bbox_left'] / float(width)
  ymin_norm = annotations['2d_bbox_top'] / float(height)
  xmax_norm = annotations['2d_bbox_right'] / float(width)
  ymax_norm = annotations['2d_bbox_bottom'] / float(height)

  difficult_obj = [0]*len(xmin_norm)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
      'image/object/class/text': dataset_util.bytes_list_feature(
          [x.encode('utf8') for x in annotations['type']]),
      'image/object/class/label': dataset_util.int64_list_feature(
          [label_map_dict[x] for x in annotations['type']]),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.float_list_feature(
          annotations['truncated']),
      'image/object/alpha': dataset_util.float_list_feature(
          annotations['alpha']),
      'image/object/3d_bbox/height': dataset_util.float_list_feature(
          annotations['3d_bbox_height']),
      'image/object/3d_bbox/width': dataset_util.float_list_feature(
          annotations['3d_bbox_width']),
      'image/object/3d_bbox/length': dataset_util.float_list_feature(
          annotations['3d_bbox_length']),
      'image/object/3d_bbox/x': dataset_util.float_list_feature(
          annotations['3d_bbox_x']),
      'image/object/3d_bbox/y': dataset_util.float_list_feature(
          annotations['3d_bbox_y']),
      'image/object/3d_bbox/z': dataset_util.float_list_feature(
          annotations['3d_bbox_z']),
      'image/object/3d_bbox/rot_y': dataset_util.float_list_feature(
          annotations['3d_bbox_rot_y']),
  }))

  return example


def filter_annotations(img_all_annotations, used_classes):
  """Filters out annotations from the unused classes and dontcare regions.
  Filters out the annotations that belong to classes we do now wish to use and
  (optionally) also removes all boxes that overlap with dontcare regions.
  Args:
    img_all_annotations: A list of annotation dictionaries. See documentation of
      read_annotation_file for more details about the format of the annotations.
    used_classes: A list of strings listing the classes we want to keep, if the
    list contains "dontcare", all bounding boxes with overlapping with dont
    care regions will also be filtered out.
  Returns:
    img_filtered_annotations: A list of annotation dictionaries that have passed
      the filtering.
  """

  img_filtered_annotations = {}

  # Filter the type of the objects.
  relevant_annotation_indices = [
      i for i, x in enumerate(img_all_annotations['type']) if x in used_classes
  ]

  for key in img_all_annotations.keys():
    img_filtered_annotations[key] = (
        img_all_annotations[key][relevant_annotation_indices])

  if 'dontcare' in used_classes:
    dont_care_indices = [i for i,
                         x in enumerate(img_filtered_annotations['type'])
                         if x == 'dontcare']

    # bounding box format [y_min, x_min, y_max, x_max]
    all_boxes = np.stack([img_filtered_annotations['2d_bbox_top'],
                          img_filtered_annotations['2d_bbox_left'],
                          img_filtered_annotations['2d_bbox_bottom'],
                          img_filtered_annotations['2d_bbox_right']],
                         axis=1)

    ious = iou(boxes1=all_boxes,
               boxes2=all_boxes[dont_care_indices])

    # Remove all bounding boxes that overlap with a dontcare region.
    if ious.size > 0:
      boxes_to_remove = np.amax(ious, axis=1) > 0.0
      for key in img_all_annotations.keys():
        img_filtered_annotations[key] = (
            img_filtered_annotations[key][np.logical_not(boxes_to_remove)])

  return img_filtered_annotations


def read_annotation_file(filename):
  """Reads a KITTI annotation file.
  Converts a KITTI annotation file into a dictionary containing all the
  relevant information.
  Args:
    filename: the path to the annotataion text file.
  Returns:
    anno: A dictionary with the converted annotation information. See annotation
    README file for details on the different fields.
  """
  with open(filename) as f:
    content = f.readlines()
  content = [x.strip().split(' ') for x in content]

  anno = {}
  anno['type'] = np.array([x[0].lower() for x in content])
  anno['truncated'] = np.array([float(x[1]) for x in content])
  anno['occluded'] = np.array([int(x[2]) for x in content])
  anno['alpha'] = np.array([float(x[3]) for x in content])

  anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
  anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
  anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
  anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])

  anno['3d_bbox_height'] = np.array([float(x[8]) for x in content])
  anno['3d_bbox_width'] = np.array([float(x[9]) for x in content])
  anno['3d_bbox_length'] = np.array([float(x[10]) for x in content])
  anno['3d_bbox_x'] = np.array([float(x[11]) for x in content])
  anno['3d_bbox_y'] = np.array([float(x[12]) for x in content])
  anno['3d_bbox_z'] = np.array([float(x[13]) for x in content])
  anno['3d_bbox_rot_y'] = np.array([float(x[14]) for x in content])

  return anno


def main(_):
  convert_kitti_to_tfrecords(
      data_dir=FLAGS.data_dir,
      output_path=FLAGS.output_path,
      classes_to_use=FLAGS.classes_to_use.split(','),
      label_map_path=FLAGS.label_map_path,
      validation_set_size=FLAGS.validation_set_size,
      train_set_size=FLAGS.train_set_size)

if __name__ == '__main__':
  tf.app.run()