from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid
from torch.nn import Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import math

class Neural_Net(Module):
	def __init__(self, in_features, embedding_size):
		super(Neural_Net, self).__init__()
		self.nn = Linear(in_features, embedding_size)
	def forward(self, img):
		out = self.nn(img)
		return out

class Neural_Net_Cosine(Module):
	def __init__(self, in_features_img, in_features_sent, embedding_size):
		super(Neural_Net_Cosine, self).__init__()
		self.nn_1 = Linear(in_features_img, 256)
		self.nn_2 = Linear(in_features_sent, 256)
		self.nn_last = Linear(256, embedding_size)
	def forward(self, img, caption):
		emb_1 = self.nn_1(img)
		emb_2 = self.nn_2(caption)
		out = self.nn_last(emb_1 + emb_2)
		return out