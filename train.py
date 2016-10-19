import argparse
import logging
import os

import mxnet as mx

from helper.config import config
from tools.train import train


def parse_args():
	parser = argparse.ArgumentParser(description='train on NUS-WIDE')
	parser.add_argument('--image_set',dest='image_set',help = 'image databese',
						default='IMGSET',type=str)
	parser.add_argument('--gpus',dest='gpu_ids',help='GPU devices',
						default='0',type=str)
	args = parser.parse_args()
	return args



if __name__ =='__main__':
	args = parse_args()
	ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
	print 'using image_set:',args.image_set
	train(config.PATH,os.getcwd())
