from data_key_points import load_test
from train_key_points import get_model, preprocess, batch_size, input_shape
from constants import key_points_desired_size
from skimage.io import imsave, imread, imshow
import numpy as np
from matplotlib import pyplot as plt

imgs, ids = load_test()
print(imgs.shape)
imgs = preprocess(imgs)
print(imgs.shape)

print('-'*30)
print('Loading saved weights...')
print('-'*30)
model = get_model(input_shape)
model.load_weights('model_kp.h5')

print('-'*30)
print('Predicting key points on test data...')
print('-'*30)
kps = model.predict(imgs, verbose=1, batch_size=batch_size)
for kp in kps:
	kp *= 16.0
	pass

np.save('kp_resutl.npy', kps)

# i = 0
# for image, image_id in zip(imgs, ids):
# 	image = (image[:, :, 0])
# 	kp = kps[i]
# 	kp *= 16

# 	middle = (kp[0], kp[1])
# 	thumb = (kp[2], kp[3])
# 	center = (kp[4], kp[5])
# 	i += 1
