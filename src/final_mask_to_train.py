# Auto select mask raw from mask-test base on mask-good-raw

import os
import shutil
from constants import mask_final_path, unet_mask_path, unet_train_path, unet_test_path

if __name__ == '__main__':
	mask_names = os.listdir(mask_final_path)

	for mask_name in mask_names:
		print('Moving ' + mask_name)
		shutil.move(os.path.join(mask_final_path,  mask_name), unet_mask_path)
		shutil.move(os.path.join(unet_test_path,  mask_name), unet_train_path)


