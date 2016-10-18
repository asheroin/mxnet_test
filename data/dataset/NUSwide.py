import os
import numpy as numpy
import cPickle
from helper.config import config
import re

class NUSwide():
	def __init__(self,img_path,root_path):
		pass
		self.name = ''
		self.cache_path = os.path.join(os.getcwd(),'cache_files')

		# type(image_set) = type('list')
		self.image_set = self.load_image_set_index()
	def load_image_set_index(self):
		# set self.image_index
		path = os.path.join(config.AnotationPath,'ImageList','ImageList.txt')
		fid = open(path,'r')

		cache_file = os.path.join(self.cache_path,self.name + 'img_set.pkl')
		# if os.path.exists(cache_file):
		# 	with open(cache_file,'rb') as fid:
		# 		image_set = cPickle.load(fid)
		# 	print 'load form exists file'
		# 	return image_set

		image_set = []
		for line in fid.readlines():
			str_ = line.split('\\')
			class_ = str_[0]
			dir_ = str_[1].split('\n')[0]
			image_set.append({'class':class_,'dir':dir_})
		pass

		with open(cache_file,'wb') as fid:
			# HIGHEST_PROTOCOL ???????
			cPickle.dump(image_set,fid,cPickle.HIGHEST_PROTOCOL)
		print 'wrote db to binary files'
		return image_set



	def load_annotation(index):
		# a simple way
		return index['class'],index['dir']
	def get_db(self):
		"""
		write something here for a better reading
		another line
		"""
		cache_file = os.path.join(self.cache_path,self.name + 'nus_db.pkl')
		# if os.path.exists(cache_file):
		# 	with open(cache_file,'rb') as fid:
		# 		nusdb = cPickle.load(fid)
		# 	print 'load form exists file'
		# 	return nusdb
			
		get_db = [self.load_annotation(index) for index in self.image_index]
		with open(cache_file,'wb') as fid:
			# HIGHEST_PROTOCOL ???????
			cPickle.dump(get_db,fid,cPickle.HIGHEST_PROTOCOL)
		print 'wrote db to binary files'

		return get_db
