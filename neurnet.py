from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid
from torch.nn import Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import math
import pdb

class Img_Emb(Module):
	def __init__(self, in_features, embedding_size):
		super(Img_Emb, self).__init__()
		self.nn = Linear(in_features, embedding_size)
	def forward(self, img):
		out = self.nn(img)
		return out


class Caption_Emb(Module):
	def __init__(self, in_features, embedding_size):
		super(Caption_Emb, self).__init__()
		self.nn = Linear(in_features, embedding_size)
	def forward(self, caption):
		out = self.nn(caption)
		return out