import argparse
import os

import mxnet as mx
from sym.vgg19 import get_vgg19_conv
from data.dataset.load_data import load_nus_db
from data.loader import NUSLoader


from pdb import set_trace as pst

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