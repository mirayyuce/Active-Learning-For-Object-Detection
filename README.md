# Active-Learning-For-Object-Detection
Active Learning For Object Detection on KITTI using MC Dropout

1. Create tfrecords: /object_detection/dataset_tools/
   prepare_kitti_splits.py
   create_kitti_tfrecord_init_and_test.py
   create_kitti_tfrecord_random.py
   

2. Train & evaluate: /object_detection/legacy/train.py and  /object_detection/legacy/eval.py

3. Config file: /object_detection/faster_rcnn_resnet101_kitti.config In this file 'use_dropout: true'
   and 'dropout_keep_probability: 0.5'

4. Export the trained graph: /object_detection/export_inference_graph.py. A flag --mc_dropout must be True. 
   With this flag the graph will keep the dropout layers for inference

5. Compute uncertainties: /object_detection/compute_image_uncertainties.py. 

6. Create tfrecords for the most informative images with /object_detection/dataset_tools/create_kitti_tfrecord_uncertainty.py
