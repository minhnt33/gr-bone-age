import json
import glob
import cv2
import numpy

annotation_path = "../mask/mask-ann/*.json"
output_path = "../mask/unet-mask/"

annotations = glob.glob(annotation_path)

for i in range(len(annotations)):
	data = json.load(open(annotations[i]))

	# The set of boundary points
	points = data["objects"][0]["points"]["exterior"]

	# opencv contour
	contour = numpy.array(points).reshape((-1, 1, 2)).astype(numpy.int32)

	img_size = data["size"]
	img_mask = numpy.zeros([img_size["height"], img_size["width"]],numpy.uint8)
	#cv2.drawContours(img_mask,[contour],0,(255,255,255),1)
	cv2.fillPoly(img_mask, pts =[contour], color=(255,255,255))
	#cv2.imshow('output', img_mask)
	saved_path = annotations[i].replace('../mask/mask-ann/', output_path).replace('.json', '.png')
	#saved_mask = cv2.resize(img_mask, (1514, 2044))
	cv2.imwrite(saved_path, img_mask)
	#cv2.waitKey(0)
	pass
