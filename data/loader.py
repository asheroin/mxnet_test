import argparse
import logging
import os

import mxnet as mx

# read settings
from helper.config import config

class NUSLoader(mx.io.DataIter):
	def __init__(self,sym,NUSdb):
		super(NUSLoader, self).__init__()

		self.sym = sym
		self.NUSdb = NUSdb

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
			self.get_bath()
			self.cur += self.batch_size
			# return a data batch
			return mx.io.DataBatch(data=_,label=_,
									pad=_,index=_,
									provide_data = self.provide_data,
									provide_label = self.provide_label)
		else 
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
		# provide data
		cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        NUSdb = [self.NUSdb[self.index[i]] for i in range(cur_from,cur_to)]
        # singel gpu
        ctx = self.ctx
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
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = minibatch.get_minibatch(iroidb, self.num_classes, self.mode)
            data_list.append(data)
            label_list.append(label)
        # reval the data list
        data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        for data, data_pad in zip(data_list,data_tensor):
        	data['data'] = data_pad[np.newaxis,:]

        new_label_list = []
        for data,label in zip(data_list,label_list):
        	data_shape = {k:v.shape for k,v in data.items()}
        	# del data_shape['im_info']
        	_,feat_shape,_ = self.feat_sym.infer_shape(**data_shape)
        	feat_shape = [int(infered) for infered in feat_shape[0]]
        	# assign anchor for label
        	label = minibatch.assign_anchor()
        	del data['im_info']
        	new_label_list.append(label)
        all_data = dict()
		for key in ['data']:
		    all_data[key] = tensor_vstack([batch[key] for batch in data_list])

	  	all_label = dict()
	  	all_label['label'] = tensor_vstack([batch['label'] for batch in new_label_list], pad=-1)


