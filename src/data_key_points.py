import os
from skimage.io import imsave, imread, imshow
from skimage.transform import resize
from matplotlib import pyplot as plt
from constants import key_points_max_size, key_points_desired_size, kp_train_path, kp_test_path, kp_json_path, constrast, kp_train_row, kp_train_col
import json
import numpy as np
from skimage.exposure import equalize_adapthist
import cv2

def preprocessing_image(image):
	max_size_img = np.zeros(key_points_max_size, dtype=np.float32)
	max_size_img[:image.shape[0], :image.shape[1]] = image
	max_size_img = resize(max_size_img, output_shape=(kp_train_row, kp_train_col), preserve_range=True)
	max_size_img /= 255
	return max_size_img

def create_train():
	# Load train images
	images = os.listdir(kp_train_path)
	total = len(images)
	kp_data = {}
	imgs = np.ndarray((total, kp_train_row, kp_train_col), dtype=np.float32)
	labels = np.ndarray((total, 6), dtype=np.uint16)

	#Load label
	with open(kp_json_path, 'r') as f:
		kp_data = json.load(f)

	ratio = 1.0 * kp_train_row / key_points_desired_size[0]
	i = 0
	for image_name in images:
		kp = np.array(kp_data[image_name], dtype=np.float32)
		kp = kp.flatten()
		kp *= ratio
		labels[i] = kp

		img = imread(os.path.join(kp_train_path, image_name), as_grey=True)
		img = preprocessing_image(img)

		img = np.array([img])
		imgs[i] = img

		# if i == 0:
		# 	radius = 5
		# 	print(kp)
		# 	cv2.circle(img, center=(kp[0], kp[1]), radius=radius, color=(255,255,255), thickness=1)
		# 	cv2.imshow(img)
		# 	cv2.waitKey(0)
			# imshow(img)
			# plt.show()

		print(image_name)
		i += 1

	np.save('kp_train.npy', imgs)
	np.save('kp_label.npy', labels)
	
	print('Saving key points train data to .npy file done.')
	pass

def create_test():
	images = os.listdir(kp_test_path)
	images.sort()
	total = len(images)

	imgs = np.ndarray((total, kp_train_row, kp_train_col), dtype=np.float32)
	imgs_id = np.ndarray((total, ), dtype=np.int32)

	i = 0
	for image_name in images:
	    img_id = int(image_name.split('.')[0])
	    img = imread(os.path.join(kp_test_path, image_name), as_grey=True)
	    img = preprocessing_image(img)
	    img = equalize_adapthist(img, clip_limit=constrast)
	    if i == 0:
	        imshow(img)
	        plt.show()
	        pass
	    img = np.array([img])
	    imgs[i] = img
	    imgs_id[i] = img_id
	    i += 1

	print('Loading done.')

	np.save('kp_imgs_test.npy', imgs)
	np.save('kp_imgs_id_test.npy', imgs_id)
	print('Saving kp test data to .npy files done.')
	pass

def load_train():
	images = np.load('kp_train.npy')
	labels = np.load('kp_label.npy')
	return images, labels

def load_test():
	imgs = np.load('kp_imgs_test.npy')
	ids = np.load('kp_imgs_id_test.npy')
	return imgs, ids

if __name__ == '__main__':
	#fix keypoint
	# data = {}
	# with open(kp_json_path, 'r') as file:
	# 	data = json.load(file)
	# #"1617.png": [[310, 64]
	# for key in data:
	# 	if key != 'current_index':
	# 		coors = data[key]
	# 		for coor in coors:
	# 			coor[0] /= 1.3
	# 			coor[1] /= 1.3

	# with open(kp_json_path, 'w') as file:
	# 	json.dump(data, file)
	create_train()
	create_test()
	pass