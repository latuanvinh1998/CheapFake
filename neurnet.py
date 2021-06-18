from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid
from torch.nn import Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import math

# class Neural_Net(Module):
# 	def __init__(self, in_features, embedding_size):
# 		super(Neural_Net, self).__init__()
# 		self.nn = Linear(in_features, embedding_size)

# 	def forward(self, img):
# 		out = self.nn(img)
# 		return out

class Neural_Net(Module):
	def __init__(self, in_features, embedding_size):
		super().__init__()
		self.nn = Linear(in_features, embedding_size)
		self.p_relu = PReLU(embedding_size)

	def forward(self, img):
		out = self.nn(img)
		out = self.p_relu(out)
		return out

class BaseLine(Module):
	def __init__(self, img_features, cap_features, embedding_size):
		super().__init__()
		self.embedding_size = embedding_size
		self.nn_img = Neural_Net(img_features, self.embedding_size)
		self.nn_cap = Neural_Net(cap_features, self.embedding_size)

	def forward(self, img, cap):
		emb_img = self.nn_img(img)
		emb_cap = self.nn_cap(cap)
		out = torch.bmm(emb_img.view(-1, 1, self.embedding_size), emb_cap.view(-1, self.embedding_size, 1))
		out = out.squeeze(-1)
		out = out.squeeze(-1)
		return out

# class Neural_Net_Cosine(Module):
# 	def __init__(self, in_features_img, in_features_sent, embedding_size):
# 		super(Neural_Net_Cosine, self).__init__()
# 		self.nn_1 = Linear(in_features_img, 256)
# 		self.nn_2 = Linear(in_features_sent, 256)
# 		self.nn_last = Linear(512, embedding_size)

# 	def forward(self, img, caption):
# 		emb_1 = self.nn_1(img)
# 		emb_2 = self.nn_2(caption)
# 		emb = torch.cat((emb_1, emb_2), 1)
		
# 		out = self.nn_last(emb)
# 		return out