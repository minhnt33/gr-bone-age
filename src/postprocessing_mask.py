import os
import json
import cv2
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage.transform import resize
from constants import mask_good_raw_path, unet_test_path, mask_final_path, desired_size

def show_image_window(image, window_name='ouput', window_size=(512, 512)):                      # Read image
	imS = cv2.resize(image, window_size)                 
	cv2.imshow(window_name, imS)                       
	cv2.waitKey(0)   
	pass

if __name__ == '__main__':
	with open('image_size.json') as data_file:
		image_size_data = json.load(data_file)

		masks = os.listdir(mask_good_raw_path)

		for mask_name in masks:
			print(mask_name)

			# Get size of original train image
			image_size = image_size_data[mask_name]
			h = image_size[0]
			w = image_size[1]

			# Load raw mask image
			img_mask = cv2.imread(os.path.join(mask_good_raw_path, mask_name), cv2.IMREAD_GRAYSCALE)
			img_mask = cv2.resize(img_mask, (desired_size, desired_size))

			# Only take portion of original train image
			img_mask = img_mask[:h, :w]

			#img_mask = cv2.fastNlMeansDenoising(img_mask, None, 10,7,21)
			#show_image_window(img_mask, window_name=mask_name, window_size=(w / 3, h / 3))

			_,img_mask = cv2.threshold(img_mask,150,255,cv2.THRESH_BINARY)
			#thresh = cv2.adaptiveThreshold(img_mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

			# Find contours
			im2, contours, hierarchy = cv2.findContours(img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

			# Find the largest contour
			largest_index = 0
			largest_area = 0
			i = -1
			for contour in contours:
				i += 1
				area = cv2.contourArea(contour)
				if area > largest_area:
					largest_area = area
					largest_index = i

			hand_contour = contours[largest_index]
 			contour_mask = np.ones(img_mask.shape[:2], dtype="uint8") * 255

 			# Draw removed contour on mask
			i = -1
			for contour in contours:
				i += 1
				if i != largest_index:
					cv2.drawContours(contour_mask, [contour], -1, 0, -1)

			# Using mask to remove undesirable contour
			img_mask = cv2.bitwise_and(img_mask, img_mask, mask=contour_mask)

 		 	#color_img = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB);
			img_mask = cv2.drawContours(img_mask, [hand_contour], -1, (255 ,255, 255), thickness=cv2.FILLED)
			# show_image_window(contour_mask, window_name='mask', window_size=(w / 3, h / 3))
			# show_image_window(img_mask, window_name=mask_name, window_size=(w / 3, h / 3))

			imsave(os.path.join(mask_final_path, mask_name), img_mask)

			cv2.destroyAllWindows()
