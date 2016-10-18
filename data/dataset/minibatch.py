
import cv2
import numpy.random as npr
import numpy as np

from helper import image_processing
from helper.config import config










def get_minibatch(db):
	num_images = len(db)
	# 随机选择缩放的目标大小
	# 这行似乎限定了是0
	random_scale_indexes = npr.randint(0, high=len(config.SCALES), size=num_images)
	im_array,im_scales = get_image_array_batch(db,config.SCALES,random_scale_indexes)
	# ......
	data = {'data':im_array}
	# label = {'label':labels_array}
	label = {}
	return data, label


def get_image_array_batch(db, scales, scale_indexes):
	num_images = len(db)
	processed_ims = []
	im_scales = []
	for i in range(num_images):
		im = cv2.read(db[i]['img_dir'])
		# target_size = scales[scale_indexes[i]]
		# since it always be a fixed value in this case
		target_size = config.SCALES[0]
		im, im_scale = image_processing.resize(im,target_size,config.MAX_SIZE)
		im_tensor = image_processing.transform(im,config.PIXEL_MEANS)
		processed_ims.append(im_tensor)
		im_scales.append(im_scale)
	array = image_processing.tensor_vstack(processed_ims)
	# return images in stacked array
	return array, im_scales