import json
import os
import numpy as np
from skimage.io import imread
from constants import unet_test_path, unet_train_path

if __name__ == '__main__':
	with open('image_size.json', 'w') as output:
		data = {}
		other_path = '../remaining-train'
		train_names = os.listdir(unet_train_path)
		test_names = os.listdir(unet_test_path)
		other_names = os.listdir(other_path)

		for train_name in train_names:
			image = imread(os.path.join(unet_train_path, train_name))
			h = image.shape[0]
			w = image.shape[1]
			data[train_name] = (h, w)

		for test_name in test_names:
			image = imread(os.path.join(unet_test_path, test_name))
			h = image.shape[0]
			w = image.shape[1]
			data[test_name] = (h, w)

		for other_name in other_names:
			image = imread(os.path.join(other_path, other_name))
			h = image.shape[0]
			w = image.shape[1]
			data[other_name] = (h, w)

		json.dump(data, output)