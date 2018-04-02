from __future__ import print_function

from skimage.transform import resize
from skimage.io import imsave, imshow
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.layers.merge import concatenate
import cv2
import matplotlib.pyplot as plt
from data_unet import load_train_data
from constants import img_cols, img_rows, get_unet

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

batch_size = 3
epochs = 300
validation_split = 0.1

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def train():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    X_train, Y_train = load_train_data()

    # imshow(X_train[0])
    # plt.show()
    # imshow(Y_train[0])
    # plt.show()

    X_train = preprocess(X_train)
    Y_train = preprocess(Y_train)

    # mean = np.mean(X_train)  # mean for data centering
    # std = np.std(X_train)  # std for data normalization
    # X_train -= mean
    # X_train /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    #model.load_weights('unet.h5')
    model.summary()
    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint('unet.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
             validation_split=validation_split,
             callbacks=[model_checkpoint, early_stopping])


    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['dice_coeff'])
    plt.plot(history.history['val_dice_coeff'])
    plt.title('model dice_coeff')
    plt.ylabel('dice_coeff')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    train()
