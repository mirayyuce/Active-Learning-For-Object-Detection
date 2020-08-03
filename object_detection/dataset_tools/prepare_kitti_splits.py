import os
import numpy as np
import argparse

def read_and_shuffle_data(path_to_images):
	images = os.listdir(path_to_images)
	shuffled_images = np.random.permutation(images)
	return shuffled_images

def prepare_splits(shuffled_images, output_path, init_size, test_size):
	init_set = shuffled_images[:init_size]
	test_set = shuffled_images[init_size:init_size+test_size]
	unlabeled_set = shuffled_images[init_size+test_size:]

	with open(output_path + "init_set.txt", "w") as output:
		for image in init_set:
			output.write("%s\n"%image)

	with open(output_path + "test_set.txt", "w") as output:
		for image in test_set:
			output.write("%s\n"%image)

	with open(output_path + "unlabeled_set.txt", "w") as output:
		for image in unlabeled_set:
			output.write("%s\n"%image)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("-k", "--PATH_TO_IMAGES", required=True, help="path to image directory /data_object_image_2/training/image_2/")
    ap.add_argument("-o", "--PATH_TO_OUTPUT", required=True, help="path to save dataset images in .txt format")
    ap.add_argument("-i", "--INIT_SIZE", required=True, type=int, default=500, help="initial set size")
    ap.add_argument("-t", "--TEST_SIZE", required=True, type=int, default=481, help="test set size")

    args = vars(ap.parse_args())

    # Read image names from the directory and return a shuffled list of images 
    shuffled_images = read_and_shuffle_data(args["PATH_TO_IMAGES"])

    # Write image names into separate .txt files for each split
    prepare_splits(shuffled_images, args["PATH_TO_OUTPUT"],args["INIT_SIZE"],args["TEST_SIZE"])

    print("Dataset splits are written to " + args["PATH_TO_OUTPUT"] + ".")
