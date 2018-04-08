# using to auto copy data from full set to kp train based on json 

import os
import shutil
import json
from constants import kp_json_path, full_data_path, kp_train_path

data_kp = {}

with open(kp_json_path, 'r') as file:
	data_kp = json.load(file)

train_images = os.listdir(kp_train_path)

image_names = data_kp.keys()
image_names.remove('current_index')
image_names.remove(train_images)

for image_name in image_names:
	shutil.copy(os.path.join(full_data_path, image_name), os.path.join(kp_train_path, image_name))
	print(image_name)
	pass