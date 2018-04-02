import numpy as np
from keras.models import Model
from data_unet import load_test_data, desired_size
from train_unet import preprocess, batch_size
import os
from skimage.transform import resize
from skimage.io import imsave
from constants import img_cols, img_rows, mask_raw_path, get_unet

print('-'*30)
print('Loading and preprocessing test data...')
print('-'*30)
imgs_test, imgs_id_test = load_test_data()
imgs_test = preprocess(imgs_test)

# mean = np.mean(imgs_test)  # mean for data centering
# std = np.std(imgs_test)  # std for data normalization
# imgs_test -= mean
# imgs_test /= std

print('-'*30)
print('Loading saved weights...')
print('-'*30)
model = get_unet()
model.load_weights('unet.h5')

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
imgs_mask_test = model.predict(imgs_test, verbose=1, batch_size=batch_size)
np.save('imgs_mask_test.npy', imgs_mask_test)

print('-' * 30)
print('Saving predicted masks to files...')
print('-' * 30)

if not os.path.exists(mask_raw_path):
    os.mkdir(mask_raw_path)

mask_size = (desired_size, desired_size)
for image, image_id in zip(imgs_mask_test, imgs_id_test):
    image = (image[:, :, 0])
    #image = resize(image, output_shape=mask_size)
    print(image_id)
    imsave(os.path.join(mask_raw_path, str(image_id) + '.png'), image)
