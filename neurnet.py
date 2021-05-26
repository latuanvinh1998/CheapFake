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

