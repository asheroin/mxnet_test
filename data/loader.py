import argparse
import logging
import os

import mxnet as mx
import numpy as np
from mxnet.executor_manager import _split_input_slice
# read settings
from helper.config import config
from data.dataset import minibatch
from helper.image_processing import tensor_vstack


# data iterator
class NUSLoader(mx.io.DataIter):
	def __init__(self,sym,NUSdb,ctx=None, work_load_list=None):
		super(NUSLoader, self).__init__()

		self.sym = sym
		self.NUSdb = NUSdb
		self.num_classes = 9
		self.batch_size = 16
		self.ctx=ctx
		if self.ctx is None:
			self.ctx = [mx.cpu()]
		self.work_load_list = work_load_list

		self.cur = 0
		self.size = len(NUSdb)

	@property
	def provide_data(self):
		return [('data',[0,0,0])]

	@property
	def provide_label(self):
		return [('label_1',shape1),
				('label_2',shape2),
				('label_3',shape3)]

	def reset(self):
		pass

	def iter_next(self):
		# bool function
		# if ( self.cur + self.batch_sie ) <= ( self.size ) 
		# return true
		return self.cur + self.batch_size <= self.size

	def next(self):
		if self.iter_next():
			self.get_batch()
			self.cur += self.batch_size
			# return a data batch
			return mx.io.DataBatch(data=self.data,label=self.label,
									pad=self.getpad(),index=self.getindex(),
									provide_data = self.provide_data,
									provide_label = self.provide_label)
		else:
			raise StopIteration

	def getindex(self):
		# return the index of current batch
		return self.cur / self.batch_size

	def getpad(self):
		# check the broader
		if self.cur+self.batch_size > size :
			return self.cur + self.batch_size - self.size
		else:
			return 0

	def get_batch(self):
		# assigin to self.data and self.label
		# pass it to mx.io.DataBatch()
		"""
		"""
		# provide data
		cur_from = self.cur
		cur_to = min(cur_from + self.batch_size, self.size)
		NUSdb = [self.NUSdb[i] for i in range(cur_from,cur_to)]
		# singel gpu
		ctx = self.ctx
		work_load_list = self.work_load_list
		if work_load_list is None:
			work_load_list = [1] * len(ctx)
		assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
		"Invalid settings for work load. "
		# for multicore-cpu?
		# is single-gpu needed?
		slices = _split_input_slice(self.batch_size, work_load_list) 
		# get mini batch
		data_list = []
		label_list = []
		for islice in slices:
			iNUSdb = [NUSdb[i] for i in range(islice.start, islice.stop)]
			data, label = minibatch.get_minibatch(iNUSdb)
			data_list.append(data)
			label_list.append(label)
		# reval the data list
		data_tensor = tensor_vstack([batch['data'] for batch in data_list])
		for data, data_pad in zip(data_list,data_tensor):
			data['data'] = data_pad[np.newaxis,:]

		new_label_list = []
		for data,label in zip(data_list,label_list):
			# every working devices
			data_shape = {k:v.shape for k,v in data.items()}
			# del data_shape['im_info']
			_,feat_shape,_ = self.sym.infer_shape(**data_shape)
			feat_shape = [int(infered) for infered in feat_shape[0]]
			# assign anchor for label
			"""
			new label or new loss
			"""
			# label = minibatch.assign_anchor()
			# del data['im_info']
			new_label_list.append(label)
		all_data = dict()

		for key in ['data']:
			all_data[key] = tensor_vstack([batch[key] for batch in data_list])

		all_label = dict()
		all_label['label'] = tensor_vstack([batch['label'] for batch in new_label_list], pad=-1)

		self.data = [mx.nd.array(all_data['data'])]
		self.label = [mx.nd.array(all_label['label'])]

