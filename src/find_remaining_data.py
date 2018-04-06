# Using to find remaining image in the data set that not belong to train set

import os
import shutil

full_data_set_path = '/root/Desktop/minh/boneage-training-dataset'
train_path = '../mask/unet-train/train'
test_path = '../mask/unet-test'

# list train file names
ignore_images = os.listdir(train_path)

# list all data set file names
full_data = os.listdir(full_data_set_path)

remaining_data = []

for image_name in full_data:
	if image_name not in ignore_images:
		remaining_data.append(image_name)

for remaining_image in remaining_data:
	shutil.copy(os.path.join(full_data_set_path, remaining_image), os.path.join(test_path, remaining_image))
	print(remaining_image)
