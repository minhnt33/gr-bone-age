from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Activation, BatchNormalization, Flatten
from data_key_points import load_train
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from keras import backend as K
from skimage.transform import resize, rotate

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

batch_size = 16
epochs = 150
validation_split = 0.1

def preprocess(imgs):
	imgs_p = np.ndarray((imgs.shape[0], 130, 100), dtype=np.float32)
	# for i in range(imgs.shape[0]):
	#     imgs_p[i] = resize(imgs[i], (130, 100), preserve_range=True)

	imgs_p = imgs_p[..., np.newaxis]
	return imgs_p

def get_model(input_shape):
	inputs = Input(shape=input_shape)

	# VGG Block 1
	block1 = Conv2D(64, (3, 3), padding='same')(inputs)
	block1 = Activation('elu')(block1)
	block1 = BatchNormalization()(block1)
	block1 = Conv2D(64, (1, 1), padding='same')(block1)
	block1 = Activation('elu')(block1)
	block1 = BatchNormalization()(block1)
	block1_pool = MaxPooling2D((3, 2), strides=(2, 2))(block1)

	# VGG Block 2
	block2 = Conv2D(128, (3, 3), padding='same')(block1_pool)
	block2 = Activation('elu')(block2)
	block2 = BatchNormalization()(block2)
	block2 = Conv2D(128, (1, 1), padding='same')(block2)
	block2 = Activation('elu')(block2)
	block2 = BatchNormalization()(block2)
	block2_pool = MaxPooling2D((3, 2), strides=(2, 2))(block2)

	# VGG Block 3
	block3 = Conv2D(256, (3, 3), padding='same')(block2_pool)
	block3 = Activation('elu')(block3)
	block3 = BatchNormalization()(block3)
	block3 = Conv2D(256, (1, 1), padding='same')(block3)
	block3 = Activation('elu')(block3)
	block3 = BatchNormalization()(block3)
	block3_pool = MaxPooling2D((3, 2), strides=(2, 2))(block3)

	drop1 = Dropout(0.5)(block3_pool)
	dense1 = Dense(512)(drop1)
	elu1 = Activation('elu')(dense1)
	drop2 = Dropout(0.5)(elu1)
	dense2 = Dense(512)(drop2)
	elu2 = Activation('elu')(dense2)
	dense3 = Dense(6)(elu2)

	model = Model(inputs=inputs, outputs=dense3)
	model.compile(optimizer='adam', loss='mean_squared_error')
	return model

def train():
	X_train, Y_train = load_train()
	X_train = preprocess(X_train)

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	model = get_model((130, 100, 1))
	model.summary()

	early_stopping = EarlyStopping(patience=10, verbose=1)
	model_checkpoint = ModelCheckpoint('model_kp.h5', monitor='val_loss', save_best_only=True)

	print('-'*30)
	print('Fitting model...')
	print('-'*30)

	history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
	     validation_split=validation_split,
	     callbacks=[model_checkpoint, early_stopping])

	print(history.history.keys())
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
	pass

if __name__ == '__main__':
	train()
	pass