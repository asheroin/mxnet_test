import argparse
import os

import mxnet as mx
from sym.vgg19 import get_vgg19_conv
from data.dataset.load_data import load_nus_db
from data.loader import NUSLoader
from mxnet.initializer import Uniform
from mxnet.module.module import Module
from pdb import set_trace as pst
from helper.config import config
from utils.load_model import load_param
from mxnet import context as cctx
import cv2

import numpy as np


def train(image_set,root_path):
	# load symbol
	# sym = get_vgg_rpn_test()

	# load training data
	NUSp, nusdb = load_nus_db(image_set,root_path)
	# data = mx.symbol.Variable(name="data")
	symbol = get_vgg19_conv(mx.symbol.Variable(name="data"))




	# build data interator
	dataIter = NUSLoader(symbol,nusdb,ctx=None, work_load_list=None)
	# infer max shape
	max_data_shape = [('data',(config.BATCH_SIZE,3,1000,1000))]
	max_data_shape_dict = {k: v for k, v in max_data_shape}
	max_label_shape = [('label',(config.BATCH_SIZE,81))]
	print 'providing maximum shape',max_data_shape,max_label_shape
	# load pretrained
	args, auxs = load_param(pretrained, epoch, convert=True)
	# initialize params

	data_names = [k[0] for k in dataIter.provide_data]
	label_names = [k[0] for k in dataIter.provide_label]
	# prepare for train
	batch_end_callback = Speedometer(train_data.batch_size, frequent=frequent)
	epoch_end_callback = mx.callback.do_checkpoint(prefix)
	# set metric

	# train





def inference(image_set,root_path,ctx=None):
	# load symbol
	# sym = get_vgg_rpn_test()

	# load training data
	NUSp, nusdb = load_nus_db(image_set,root_path)
	data = mx.symbol.Variable(name="data")
	symbol = get_vgg19_conv(mx.symbol.Variable(name="data"))
	# mod = mx.model.FeedForward.load('vgg',1,ctx=mx.gpu(),numpy_batch_size=config.BATCH_SIZE)
	
	

	
	# build data interator
	dataIter = NUSLoader(nusdb,ctx=[mx.gpu(0)], work_load_list=None)
	# infer max shape
	max_data_shape = [('data',(config.BATCH_SIZE,3,600,600))]
	max_data_shape_dict = {k: v for k, v in max_data_shape}
	max_label_shape = [('relu5_3',(config.BATCH_SIZE,512))]
	print 'providing maximum shape',max_data_shape,max_label_shape
	# load pretrained
	pretrained = 'vgg16'
	epoch = 1
	arg_params, aux_params = load_param(pretrained, epoch, convert=True)
	# pst()
	# bind / memory allocate
	mod = Module(symbol,context=cctx.gpu())
	# mod.bind(max_data_shape,max_label_shape)
	mod.bind(data_shapes=max_data_shape)
	# initialize params

	mod.init_params(initializer=Uniform(0.01), arg_params=arg_params,
							aux_params=aux_params, allow_missing=False,
							force_init=False)
	# executor = symbol.bind(ctx,arg_params, args_grad=None,
	# 					grad_req='null', aux_states=self.aux_params)
	# data_names = [k[0] for k in dataIter.provide_data]
	# label_names = [k[0] for k in dataIter.provide_label]
	# prepare for train
	# batch_end_callback = Speedometer(train_data.batch_size, frequent=frequent)
	# epoch_end_callback = mx.callback.do_checkpoint(prefix)
	# set metric

	# for nb,evalb in enumerate(dataIter):
	# 	arg_params['data'] = evalb.data[0].asnumpy()

	# 	# mod.forward(evalb,is_train=False)
	# 	executor = symbol.bind([mx.gpu(0)],arg_params, args_grad=None,grad_req='null', aux_states=aux_params)
	# 	executor.forward(is_train=False)
	# 	pst()
	# mod.predict(dataIter)

	epoch = 200
	cp_count = 0
	count = 0
	res = []



	for preds,i_batch,batch in mod.iter_predict(dataIter):
		pred_label = preds[0].asnumpy().argmax(axis=1)
		print 'finished %d @ saved %d...\r'%(count,cp_count),
		res.append(pred_label)
		count+=1
		if count==epoch:
			np.save('extrac\\feature-%04d.npy'%cp_count,res)
			cp_count+=1
			count=0
			res = []
	if len(res)!=0:
		np.save('extrac\\feature-%04d.npy'%cp_count,res)


