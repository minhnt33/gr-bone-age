# Using to find remaining image in the data set that not belong to train set and move them to test folder

import os
import shutil
from constants import full_data_path, unet_train_path, unet_test_path, kp_train_path, kp_test_path
import sys

option = sys.argv[1]
train_path = ''
test_path = ''

if option == 'unet':
	train_path = unet_train_path
	test_path = unet_test_path
elif option == 'kp':
	train_path = kp_train_path
	test_path = kp_test_path
else:
	print('Please use argv unet or kp')

if train_path != '':
	# list train file names
	ignore_images = os.listdir(train_path)

	# list all data set file names
	full_data = os.listdir(full_data_path)

	remaining_data = []

	for image_name in full_data:
		if image_name not in ignore_images:
			remaining_data.append(image_name)

	for remaining_image in remaining_data:
		shutil.copy(os.path.join(full_data_path, remaining_image), os.path.join(test_path, remaining_image))
		print(remaining_image)