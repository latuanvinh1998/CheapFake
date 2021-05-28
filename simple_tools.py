import numpy as np
import math
import torch
from torch import nn

def resize_bb(arr, width, height):

	size = (arr[2] - arr[0]) - (arr[3] - arr[1])

	if size < 0:
		if arr[0] - abs(size)/2 >=0 and arr[2] + abs(size)/2 < width:
			arr[0] -= abs(size)/2
			arr[2] += abs(size)/2

	elif size > 0:
		if arr[1] - abs(size)/2 >=0 and arr[3] + abs(size)/2 < height:
			arr[1] -= abs(size)/2
			arr[3] += abs(size)/2
	
	return arr.astype(int)

def E1(TP, TN, FP, FN):

	accuracy = (TP + TN)/(TP + FP + FN + TN)
	precision = TP/(TP + FP)
	recall = TP/(TP + FN)
	f1_score = 2*(recall*precision)/(recall + precision)
	mcc = (TP * TN - FP * FN)/math.sqrt ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

	return accuracy, f1_score, mcc

def select_triplets(anchor, positive, negative):
	triplet = []
	dist = []

	for i in range(len(anchor)):
		# dist_p = torch.dist(anchor[i], positive, 2).detach().cpu().numpy()
		# dist_n = torch.dist(anchor[i], negative, 2).detach().cpu().numpy()
		dist.append(triplet_loss(anchor[i], positive, negative))
	print(dist)

def extract(model_ef, img):
	with torch.no_grad():
		feature = model_ef.extract_features(img.to(torch.device("cuda:0")))
		feature = nn.AdaptiveAvgPool2d(1)(feature)
		feature = torch.squeeze(feature, -1)
		feature = torch.squeeze(feature, -1)
	return feature