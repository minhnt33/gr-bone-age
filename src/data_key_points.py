import os
from skimage.io import imsave, imread, imshow
from skimage.transform import resize
from matplotlib import pyplot as plt
from constants import key_points_max_size
import json
import numpy as np

kp_json_path = '../key-points/key_points.json'
kp_train_path = '../key-points/train'
kp_test_path = '../key-points/test'
img_train_size = (130, 100)

def preprocessing_image(image):
	max_size_img = np.zeros(key_points_max_size, dtype=np.float32)
	max_size_img[:image.shape[0], :image.shape[1]] = image
	max_size_img = resize(max_size_img, output_shape=img_train_size, preserve_range=True)
	max_size_img /= 255
	return max_size_img

def create_train():
	# Load train images
	images = os.listdir(kp_train_path)
	total = len(images)
	kp_data = {}
	imgs = np.ndarray((total, img_train_size[0], img_train_size[1]), dtype=np.float32)
	labels = np.ndarray((total, 6), dtype=np.uint16)

	#Load label
	with open(kp_json_path, 'r') as f:
		kp_data = json.load(f)

	i = 0
	for image_name in images:
		img = imread(os.path.join(kp_train_path, image_name), as_grey=True)
		img = preprocessing_image(img)
		# if i == 0:
		# 	imshow(img)
		# 	plt.show()

		img = np.array([img])
		imgs[i] = img

		kp = np.array(kp_data[image_name])
		labels[i] = kp.flatten()
		print(image_name)
		i += 1

	np.save('kp_train.npy', imgs)
	np.save('kp_label.npy', labels)
	
	print('Saving key points train data to .npy file done.')
	pass

def create_test():
	images = os.listdir(kp_test_path)

	i = 0
	for image_name in images:
	    img = imread(os.path.join(kp_test_path, image_name), as_grey=True)
	    img = preprocessing_image(img)
	    if i == 0:
	        imshow(img)
	        plt.show()
	        pass
	    img = np.array([img])
	    imgs[i] = img
	    i += 1

	print('Loading kp test images done.')

	np.save('kp_test.npy', imgs)
	print('Saving key points test data to .npy file done.')
	pass

def load_train():
	images = np.load('kp_train.npy')
	labels = np.load('kp_label.npy')
	return images, labels

def load_test():
	pass

if __name__ == '__main__':
	create_train()
	#create_test()
	pass