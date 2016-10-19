import numpy as np
from easydict import EasyDict as edict

config = edict()

# edit settings

# for examples
config.PATH = 'D:\Flickr\Flickr'
config.AnotationPath = 'F:\NUSWIDE'

config.SCALES = (600, )  # single scale training and testing
config.MAX_SIZE = 1000
config.PIXEL_MEANS = np.array([[[123.68, 116.779, 103.939]]])