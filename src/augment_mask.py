from data_mask import load_train_data
from constants import augmented_train_path, unet_train_path, unet_mask_path, augmented_sample_amount, desired_size
from unetdatagen import ImageDataGenerator
from train_unet import batch_size
import os
import shutil
import numpy as np
from skimage.transform import resize

def preprocess(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def create_augmented_data():
	X_train, Y_train = load_train_data()

	X_train = preprocess(X_train, desired_size, desired_size)
	Y_train = preprocess(Y_train, desired_size, desired_size)

	train_generator = ImageDataGenerator(
							# featurewise_std_normalization=True,
							# zca_whitening=True,
							# featurewise_center=True,
							# rotation_range=45,
	                        width_shift_range=0.2,
	                        height_shift_range=0.2,
	                        # shear_range=0.1,
	                        # zoom_range=0.1,
	                        # horizontal_flip=True,
	                        # vertical_flip=True,
	                        fill_mode='constant',
	                        cval=0)

	# X_train = train_generator.fit(X_train)

	aug_data = train_generator.flow(X=X_train, y=Y_train, batch_size=batch_size, save_to_dir=augmented_train_path, save_prefix='aug', save_format='png')

	i = 0
	for batch in aug_data:
		i += 1
		print('Saving ' + str(i))
		if i > augmented_sample_amount:
			print('Saving augmented train data done')
			break

def move_augmented_data():
	aug_names = os.listdir(augmented_train_path)

	for aug_name in aug_names:
		# If it is mask
		if 'mask' in aug_name:
			shutil.move(os.path.join(augmented_train_path,  aug_name), os.path.join(unet_mask_path,  aug_name))
		else: # If it is image
			shutil.move(os.path.join(augmented_train_path,  aug_name), os.path.join(unet_train_path,  aug_name))
	print('Moving augmented data done')

if __name__ == '__main__':
	#create_augmented_data()
	move_augmented_data()
