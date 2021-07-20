from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid
from torch.nn import Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import math


class Neural_Net_Triplet(Module):
	def __init__(self, in_features_img, in_features_sent, embedding_size):
		super(Neural_Net_Triplet, self).__init__()
		self.nn_1 = Linear(in_features_img, 512)
		self.nn_2 = Linear(in_features_sent, 512)
		self.prelu = PReLU(512)
		self.nn_last = Linear(512, embedding_size)

	def forward(self, img, caption):
		emb_1 = self.nn_1(img)
		#emb_1 = self.prelu_1(emb_1)

		emb_2 = self.nn_2(caption)
		#emb_2 = self.prelu_2(emb_2)
		
		emb = emb_1 + emb_2
		emb = self.prelu(emb)
		
		out = self.nn_last(emb)
		return out

class Neural_Net_Cosine(Module):
	def __init__(self, in_features_img, in_features_sent, embedding_size):
		super(Neural_Net_Cosine, self).__init__()
		self.nn_1 = Linear(in_features_img, 512)
		self.nn_2 = Linear(in_features_sent, 512)
		self.prelu_1 = PReLU(512)
		self.prelu_2 = PReLU(512)
		self.nn_last = Linear(512, embedding_size)

	def forward(self, img, caption_1):
		emb_1 = self.nn_1(img)
		emb_1 = self.prelu_1(emb_1)

		emb_2 = self.nn_2(caption_1)
		emb_2 = self.prelu_2(emb_2)	
		emb = emb_1 + emb_2
		
		out = self.nn_last(emb)
		return out

class Neural_Net(Module):
	def __init__(self, in_features_img, in_features_sent):
		super(Neural_Net, self).__init__()
		self.nn_1 = Linear(in_features_img, 512)
		self.nn_2 = Linear(in_features_sent, 512)
		self.prelu = PReLU(512)
		self.nn_last = Linear(512, 2)

	def forward(self, img, caption_1, caption_2):
		emb_1 = self.nn_1(img)

		emb_2 = self.nn_2(caption_1)

		emb_3 = self.nn_2(caption_2)

		emb = emb_1 + emb_2 + emb_3
		emb = self.prelu(emb)

		out = self.nn_last(emb)
		return out
