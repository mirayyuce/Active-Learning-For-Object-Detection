import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import tensorflow as tf
import copy
import csv
from sklearn.cluster import DBSCAN
import argparse
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from PIL import Image
from scipy import stats
import cv2
import json
import gc

#tf.logging.set_verbosity(tf.logging.ERROR)
sys.path.append("..")


def load_detection_graph(PATH_TO_FROZEN_GRAPH):
  #get graph
  detection_graph = tf.Graph()

  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()

    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  return detection_graph

def train_dbscan(X_train):
    clustering = DBSCAN(min_samples=10, eps=0.05)

    y_train = clustering.fit_predict(X_train)
    
    return y_train

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def get_cluster_dict(y_train):

  cluster_dict = {}
  for ind, cluster in enumerate(y_train):

    if cluster in cluster_dict:
      cluster_dict[cluster].append(ind)
    else:
      cluster_dict[cluster] = [ind]

  return cluster_dict

def compute_cluster_uncertainties(flat_boxes, flat_softmax):

  uncertainties_cluster_entropy_sum_var = []
  uncertainties_cluster_entropy_avg_var = []

  y_train = train_dbscan(flat_boxes)

  cluster_dict = get_cluster_dict(y_train)

  for cls_id, cluster in enumerate(cluster_dict): 
    indices = cluster_dict[cluster]          
    boxes_cls = []
    softmax = []
    
    # get scores and categories of the cluster members
    for ind in indices:
      # scores.append(final_output['detection_scores'][ind])
      boxes_cls.append(flat_boxes[ind])
      # classes.append(final_output['detection_classes'][ind])
      softmax.append(flat_softmax[ind])
    
    # classification uncertainty
    avg_softmax = np.mean(softmax, axis = 0)
    
    entropy = stats.entropy(avg_softmax, base=2)

    # localization uncertainty

    top_left = np.array(boxes_cls)[:,:2]
    right_bottom = np.array(boxes_cls)[:,2:]

    var_top_left = np.var(top_left)
    var_right_bottom = np.var(right_bottom)

    avg_var = (var_top_left + var_right_bottom)/2.0

    # cluster uncertainty
    uncertainties_cluster_entropy_avg_var.append(avg_var + entropy)

  return uncertainties_cluster_entropy_avg_var


def run_inference_for_images(images, graph):
  with graph.as_default():
    with tf.Session() as sess:
      output_dict_array = []

      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}

      for key in ['detection_boxes','detection_scores', "SecondStagePostprocessor/Reshape_6"]:
      
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)


      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      
      image_uncertainty_list = []
      
      for img_ind, img_path in enumerate(images):
        image_dict = {}
        image = Image.open(img_path)
        print("\r",img_ind,end=" ")

        image_np = load_image_into_numpy_array(image)

        final_output = {}
        for key in ['detection_boxes', "SecondStagePostprocessor/Reshape_6"]:  
          final_output[key] = []
       
        for drop in range(50):

          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image_np, 0)})
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          output_dict['SecondStagePostprocessor/Reshape_6'] = output_dict['SecondStagePostprocessor/Reshape_6'][0]

          # get non-trivial detections < 0.1
          for ind, score in enumerate(output_dict['detection_scores']):
            if score >= 0.1:
              
              final_output['detection_boxes'].append(output_dict['detection_boxes'][ind])
              final_output['SecondStagePostprocessor/Reshape_6'].append(output_dict['SecondStagePostprocessor/Reshape_6'][ind])

        del image, image_np
        gc.collect()
        flat_boxes = final_output['detection_boxes'] #[item for sublist in final_output['detection_boxes'] for item in sublist]
        flat_softmax = final_output['SecondStagePostprocessor/Reshape_6'] #[item for sublist in final_output['SecondStagePostprocessor/Reshape_6'] for item in sublist]
        
        if len(flat_boxes) == 0:

          image_dict["image_path"] = img_path
          
          image_dict["AVG"] = 0.0
          image_dict["TOTAL"] = 0.0

          image_uncertainty_list.append(image_dict) 

          del flat_boxes, flat_softmax, final_output

          continue
        else:
    
          uncertainties_cluster_entropy_avg_var = compute_cluster_uncertainties(flat_boxes, flat_softmax)

          del flat_boxes, flat_softmax, final_output

          image_dict["image_path"] = img_path
          # raw uncertainty
          image_dict["uncertainties"] = uncertainties_cluster_entropy_avg_var
          # image's uncertainty metric 1
          image_dict["AVG"] = float(np.average(uncertainties_cluster_entropy_avg_var)) 
          # image's uncertainty metric 2
          image_dict["TOTAL"] = float(np.sum(uncertainties_cluster_entropy_avg_var))  

          image_uncertainty_list.append(image_dict)



  return image_uncertainty_list

def get_images(kitti_dir, unlabeled_dir, round_number):
  images_tmp = []
  image_paths = []

  # Read image names to a list 
  with open(unlabeled_dir, "r") as f:
    images_tmp = f.readlines()
  f.close()

  # Retrieve absolute paths to the active learning round images
  for img in images_tmp[round_number * 1300:(round_number + 1) * 1300]:
    image_paths.append(os.path.join(kitti_dir, img[:-1]))

  # Return absolute paths of the active learning round images
  return image_paths


def prepare_sorted_results(output_dir, round_number, metric):
    all_results = []
    path_to_save_json = args["PATH_TO_OUTPUT"] + "round_" + str(args["ROUND"]) + "_unsorted.json"


    with open(path_to_save_json) as read:
        all_results = json.load(read)

    if metric:
      style = "TOTAL"
    else:
      style = "AVG"

    all_results.sort(key=lambda x: x[style], reverse = True)


    path_to_save_sorted_json = args["PATH_TO_OUTPUT"] + "round_" + style + "_" + str(args["ROUND"]) + "_sorted.json"
    path_to_save_sorted_txt = args["PATH_TO_OUTPUT"] + "round_" + style + "_" + str(args["ROUND"]) + "_sorted.txt"
    

    with open(path_to_save_sorted_json, "w") as test:
      json.dump(all_results, test)

    with open(path_to_save_sorted_txt, "w") as test:

        for entry in all_results:
            
            test.write(str(entry["image_path"]) +" "+ str(entry[style])+"\n")

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("-g", "--PATH_TO_FROZEN_GRAPH", required=True, help="path to frozen detection graph frozen_inference_graph.pb")
    ap.add_argument("-o", "--PATH_TO_OUTPUT", required=True, help="path to json files to store all uncertainty values")
    ap.add_argument("-k", "--PATH_TO_KITTI_DIR", required=True, help="path to image directory /data_object_image_2/training/image_2/")
    ap.add_argument("-i", "--PATH_TO_IMAGES_LIST", required=True, help="path to the list of shuffled unlabeled images unlabaled_set.txt")
    ap.add_argument("-r", "--ROUND", type=int, default=0, help="current active learning round, [0,4].")
    #ap.add_argument("-m", "--METRIC", type=int, default=0, help="metric to compute informativeness. 0 is AVG, 1 is TOTAL")

    args = vars(ap.parse_args())

    # Absolute paths to current round's unlabeled images
    images = get_images(args["PATH_TO_KITTI_DIR"], args["PATH_TO_IMAGES_LIST"], args["ROUND"])
    
    # Load frozen detection graph to the memory
    detection_graph = load_detection_graph(args["PATH_TO_FROZEN_GRAPH"])

    # Informativeness values for all unlabeled images
    image_uncertainty_list = run_inference_for_images(images, detection_graph)

    path_to_save_json = args["PATH_TO_OUTPUT"] + "round_" + str(args["ROUND"]) + "_unsorted.json"

    with open(path_to_save_json, "w") as output:
      json.dump(image_uncertainty_list, output)

    prepare_sorted_results(args["PATH_TO_OUTPUT"], args["ROUND"], "AVG")
    prepare_sorted_results(args["PATH_TO_OUTPUT"], args["ROUND"], "TOTAL")
    