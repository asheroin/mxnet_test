import os
import numpy as numpy
import cPickle
from helper.config import config
import re


import pdb



# class of database
class NUSwide():
	def __init__(self,img_path,root_path):
		pass
		self.name = ''
		self.root_path = root_path
		self.cache_path = os.path.join(root_path,'cache_files')

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
		class2id = {}
		for index,line in enumerate(fid.readlines()):
			str_ = line.split('\\')
			class_ = str_[0]
			class2id[class_] = 0
			dir_ = str_[1].split('\n')[0]

			image_set.append({'class':class_,'dir':dir_,'index':index,'index_of_class':dir_.split('_')[0]})
		pass
		pdb.set_trace()
		
		for index,value in enumerate(class2id):
			class2id[value] = index

		for item in image_set:
			item['classId']=class2id[item['class']]

		with open(cache_file,'wb') as fid:
			# HIGHEST_PROTOCOL ???????
			cPickle.dump(image_set,fid,cPickle.HIGHEST_PROTOCOL)
		print 'wrote db to binary files'

		return image_set



	def load_annotation(self,item):
		# a simple way
		return {'class':item['class'],'classId':item['classId'],'index':item['index'],'index_of_class':item['index_of_class'],'dir':item['dir'],'file_path':os.path.join(config.PATH,item['class'],item['dir'])}
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
			
		get_db = [self.load_annotation(item) for item in self.image_set]
		with open(cache_file,'wb') as fid:
			# HIGHEST_PROTOCOL ???????
			cPickle.dump(get_db,fid,cPickle.HIGHEST_PROTOCOL)
		print 'wrote db to binary files'

		return get_db
