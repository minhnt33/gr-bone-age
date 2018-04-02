import Tkinter as tk
from PIL import Image, ImageTk
from constants import key_points_in_path, color_red, color_yellow, color_blue
import os
import os.path
from skimage.io import imread
from Tkinter import Frame, SUNKEN, Scrollbar, Canvas, HORIZONTAL, E, S, N, W, BOTH, ALL
from constants import key_points_max_size, key_points_desired_size, key_points_display_size
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import json

current_index = -1
current_image_name = 'None'
key_points_data = {}
tip_middle_finger_point = None # yellow
tip_thumb_point = None # blue
center_capitate_point = None # red
point_count = 0
ouput_path = '../key-points/key_points.json'

def next_image_path():
	global current_index
	global current_image_name
	current_index += 1
	current_image_name = image_names[current_index]
	root.title(current_image_name)
	return os.path.join(key_points_in_path, current_image_name)

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

# Check output file
if os.path.isfile(ouput_path):
	with open(ouput_path, 'r') as file:
		key_points_data = json.load(file)
	current_index = key_points_data['current_index']

# Get all image name
image_names = os.listdir(key_points_in_path)
image_names.sort()

root = tk.Tk()

if len(image_names) == 0:
	print(key_points_in_path + ' is empty')
else:
	# Read image and preprocessing
	img_data = imread(next_image_path(), as_grey=True)
	img_data = preprocessing_image(img_data)
	current_image = ImageTk.PhotoImage(img_data)

	canvas = tk.Canvas(root, width=img_data.size[0], height=img_data.size[1])
	canvas_img = canvas.create_image(0, 0, image=current_image, anchor="nw")
	canvas.pack(side = "bottom", fill = "both", expand = "yes")

	# ratio used to convert coordinate to desired size
	ratio = key_points_desired_size[0] / img_data.size[0]

	# Function to be called when mouse is clicked
	def printcoords(event):
		print(event.x,event.y)
		global current_image_name
		global tip_middle_finger_point
		global tip_thumb_point
		global center_capitate_point
		global point_count
		global key_points_data

		# Init keypoint array for this image
		if current_image_name not in key_points_data:
			key_points_data[current_image_name] = []

		point_count += 1

		key_points_data[current_image_name].append((event.x * ratio, event.y * ratio))
		if point_count == 1:
			#print ('Middle Finger Tip')
			tip_middle_finger_point = paint_dot(event.x, event.y, color_yellow) 
		elif point_count == 2:
			#print ('Thumb Tip')
			tip_thumb_point = paint_dot(event.x, event.y, color_blue)
		elif point_count == 3:
			#print ('Center Capitate')
			center_capitate_point = paint_dot(event.x, event.y, color_red)

	# Show next image when press Space
	def show_next_image(event):
		if current_index < len(image_names) - 1:
			global point_count
			global current_image # Need to maintain reference to avoid garbage collecting
			global img_data
			point_count = 0
			canvas.delete(tip_middle_finger_point)
			canvas.delete(tip_thumb_point)
			canvas.delete(center_capitate_point)
			img_data = imread(next_image_path(), as_grey=True)
			img_data = preprocessing_image(img_data)
			current_image = ImageTk.PhotoImage(img_data)
			canvas.itemconfig(canvas_img, image=current_image)
		else:
			print("Done")

    #mouseclick event
	canvas.bind("<Button 1>",printcoords)

	# Next image
	root.bind("<space>", show_next_image)

	root.mainloop()

	# Save current image index to revisit later
	key_points_data['current_index'] = current_index

	# Save to json
	with open(ouput_path, 'w') as output:
		json.dump(key_points_data, output)