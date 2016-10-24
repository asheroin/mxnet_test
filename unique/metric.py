import mxnet as mx 
import numpy as np 

from helper.config import config


class selfMetric(mx.metric.EvalMetric):
	def __init__(self):
		super(selfMetric,self).__init__('selfMetric')
	def updata(self,labels,preds):

		last_dim = 


		self.sum_metric = []
		self.num_inst = []