from __future__ import print_function
import matplotlib
#matplotlib.use('Agg') # Set to prevent matplotlib from using the Xwindows backend on server

import os
import numpy as np
from skimage.io import imsave, imread, imshow
from skimage.transform import resize
from skimage.exposure import equalize_adapthist
from matplotlib import pyplot as plt
import math
from constants import img_rows, img_cols, unet_train_path, unet_test_path, unet_mask_path, desired_size, unet_mask_val_path, unet_train_val_path, constrast

def make_square_image(img, desired_size=128):
    h = float(img.shape[0])
    w = float(img.shape[1])
    if h > desired_size or w > desired_size:
        ratio = 1.0 * (h / desired_size) if (h >= w) else 1.0 * (w / desired_size)
        h /= ratio
        w /= ratio
        h = int(math.floor(h))
        w = int(math.floor(w))
        img = resize(img, output_shape=(h, w), preserve_range=True)

    result = np.zeros((desired_size, desired_size), dtype=np.float32)
    result[:img.shape[0], :img.shape[1]] = img
    return result

def create_train_data():    
    images = os.listdir(unet_train_path)
    masks = os.listdir(unet_mask_path)
    images.sort() # Very important to sort the file before processing. It helps maintain pair between image and mask
    masks.sort()
    total = len(images)

    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.float32)

    i = 0
    for image_name in images:
        img = imread(os.path.join(unet_train_path, image_name), as_grey=True)

        if img.shape[0] != img.shape[1]:
            img = make_square_image(img, desired_size=desired_size)
            pass
        
        img = resize(img, output_shape=(img_rows, img_cols), preserve_range=True)
        img /= 255.0
        #img = equalize_adapthist(img, clip_limit=constrast)

        #if i == 0:
        #    imshow(img)
        #    plt.show()
        #    pass
        img = np.array([img], dtype=np.float32)
        imgs[i] = img
        print(image_name)
        i += 1

    np.save('imgs_train.npy', imgs)
    print('Saving train image to .npy done.')

    i = 0
    for mask_name in masks:
        img_mask = imread(os.path.join(unet_mask_path, mask_name), as_grey=True)
	
        if img_mask.shape[0] != img_mask.shape[1]:
            img_mask = make_square_image(img_mask, desired_size=desired_size)
            pass        
            
        img_mask = resize(img_mask, output_shape=(img_rows, img_cols), preserve_range=True)
        img_mask /= 255.0

        #if i == 0:
        #    imshow(img_mask)
        #    plt.show()
        #    pass
        img_mask = np.array([img_mask], dtype=np.float32)
        imgs_mask[i] = img_mask
        print('Mask ' + mask_name)
        i += 1

    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving train mask to .npy done.')

def create_test_data():
    images = os.listdir(unet_test_path)
    images.sort()
    total = len(images)

    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(unet_test_path, image_name), as_grey=True)
	
        h = float(img.shape[0])
        w = float(img.shape[1])
        if h > desired_size or w > desired_size:
            ratio = 1.0 * (h / desired_size) if (h >= w) else 1.0 * (w / desired_size)
            h /= ratio
            w /= ratio
            h = int(math.floor(h))
            w = int(math.floor(w))
            img = resize(img, output_shape=(h, w), preserve_range=True)

        img = make_square_image(img, desired_size=desired_size)
        img = resize(img, output_shape=(img_rows, img_cols), preserve_range=True)	
        img /= 255.0
        img = equalize_adapthist(img, clip_limit=constrast)
        #if i == 0:
        #    imshow(img)
        #    plt.show()
        #    pass
        img = np.array([img], dtype=np.float32)
        imgs[i] = img
        imgs_id[i] = img_id
        print(image_name)
        i += 1

    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving test data to .npy files done.')

def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()
