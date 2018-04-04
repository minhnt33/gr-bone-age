import Tkinter as tk
from PIL import Image, ImageTk
from constants import kp_test_path, color_red, color_yellow, color_blue
import os
import os.path
from skimage.io import imread
from Tkinter import Frame, SUNKEN, Scrollbar, Canvas, HORIZONTAL, E, S, N, W, BOTH, ALL
from constants import key_points_max_size, key_points_desired_size, key_points_display_size, kp_train_path, kp_json_path
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import json

current_index = -1
current_image_name = 'None'
tip_middle_finger_point = None # yellow
tip_thumb_point = None # blue
center_capitate_point = None # red
ouput_path = '../key-points/key_points.json'
# ratio used to convert coordinate to display size
ratio = 1.0 * key_points_display_size[0] / key_points_desired_size[0]

def next_image_path():
	global current_index
	global current_image_name
	current_index += 1
	current_image_name = image_names[current_index]
	root.title(current_image_name)
	return os.path.join(kp_train_path, current_image_name)

def preprocessing_image(image):
	max_size_img = np.zeros(key_points_max_size, dtype=np.float32)
	max_size_img[:image.shape[0], :image.shape[1]] = image
	max_size_img = resize(max_size_img, output_shape=key_points_display_size, preserve_range=True)
	max_size_img /= 255
	return Image.fromarray(np.uint8(max_size_img*255))

# Paint dot
def paint_dot(x, y, color):
    x1, y1 = (x - 1), (y - 1)
    x2, y2 = (x + 1), (y + 1)
    return canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color, width=10)

# Get all image name
image_names = os.listdir(kp_train_path)
image_names.sort()

# Load kp
kps = {}
with open(kp_json_path, 'r') as file:
	kps = json.load(file)

root = tk.Tk()

if len(image_names) == 0:
	print(kp_test_path + ' is empty')
else:
	# Read image and preprocessing
	img_data = imread(next_image_path(), as_grey=True)
	img_data = preprocessing_image(img_data)
	current_image = ImageTk.PhotoImage(img_data)

	canvas = tk.Canvas(root, width=img_data.size[0], height=img_data.size[1])
	canvas_img = canvas.create_image(0, 0, image=current_image, anchor="nw")
	canvas.pack(side = "bottom", fill = "both", expand = "yes")

	kp = kps[current_image_name]
	kp = np.array(kp, dtype=np.float32)
	kp *= ratio
	print(kp)
	# Paint dots
	tip_middle_finger_point = paint_dot(kp[0], kp[1], color_yellow)
	tip_thumb_point = paint_dot(kp[2], kp[3], color_blue)
	center_capitate_point = paint_dot(kp[4], kp[5], color_red)

	# Show next image when press Space
	def show_next_image(event):
		if current_index < len(image_names) - 1:
			global tip_middle_finger_point
			global tip_thumb_point
			global center_capitate_point
			global current_image # Need to maintain reference to avoid garbage collecting
			global img_data
			canvas.delete(tip_middle_finger_point)
			canvas.delete(tip_thumb_point)
			canvas.delete(center_capitate_point)
			img_data = imread(next_image_path(), as_grey=True)
			img_data = preprocessing_image(img_data)
			current_image = ImageTk.PhotoImage(img_data)
			canvas.itemconfig(canvas_img, image=current_image)

			kp = kps[current_image_name]
			kp = np.array(kp, dtype=np.float32)
			kp *= ratio
			print(kp)
			# Paint dots
			tip_middle_finger_point = paint_dot(kp[0], kp[1], color_yellow)
			tip_thumb_point = paint_dot(kp[2], kp[3], color_blue)
			center_capitate_point = paint_dot(kp[4], kp[5], color_red)
		else:
			print("Done")

	# Next image
	root.bind("<space>", show_next_image)

	root.mainloop()