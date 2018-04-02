from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D

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
	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
	return model

def train():
	
	pass