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
	
	# dataIter.get_batch()

	pst()
    # build data interator
	dataIter = NUSLoader(symbol,nusdb,ctx=None, work_load_list=None)
    # infer max shape

    # load pretrained

    # initialize params

    # prepare for train, such as setting metric

    # train