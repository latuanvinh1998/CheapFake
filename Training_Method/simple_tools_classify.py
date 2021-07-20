import numpy as np

import math
import torch
import random
import difflib

from PIL import Image
from torchvision import transforms
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

	return accuracy, f1_score, mcc, precision, recall


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


def get_pair_cap(idx, length, labels, i):

	cap_1 = labels[idx]["articles"][0]['caption_modified'].replace('\n','')

	if len(labels[idx]["articles"]) > 1 and i <= 5:
		for j in range(len(labels[idx]["articles"])):
			if j != len(labels[idx]["articles"]) - 1:
				cap_2 = labels[idx]["articles"][j + 1]['caption_modified'].replace('\n','')
				if difflib.SequenceMatcher(None, cap_1, cap_2).ratio() < 0.5:
					return cap_1, cap_2, 0

			else:
				r = random.choice([k for k in range(0,length) if k not in [idx]])
				cap_2 = labels[r]["articles"][random.randint(0, len(labels[r]["articles"]) - 1)]['caption_modified'].replace('\n','')
				return cap_1, cap_2, 1

	else:
		r = random.choice([k for k in range(0,length) if k not in [idx]])
		cap_2 = labels[r]["articles"][random.randint(0, len(labels[r]["articles"]) - 1)]['caption_modified'].replace('\n','.')
		return cap_1, cap_2, 1


def get_emb_batch(model_clip, imgs, sent_1, sent_2):

	batch_img = torch.stack([img for img in imgs])

	with torch.no_grad():
		features = model_clip.encode_image(batch_img.to(torch.device("cuda:0"))).type(torch.cuda.FloatTensor)

	emb_batch_1 = torch.stack([emb for emb in sent_1])
	emb_batch_2 = torch.stack([emb for emb in sent_2])

	return features, emb_batch_1, emb_batch_2


def validation_loss(model_bert, model_clip, neural, labels, criterion, preprocess):

	# length = len(labels)
	random.shuffle(labels)
	length = 1600
	iters = int(length/8)
	loss = 0

	neural.eval()

	for k in range(iters):

		imgs = []
		sent_1 = []
		sent_2 = []
		target = []

		for i in range(8):

			idx = 8*k + i
			
			path = '../Data/' +  labels[i]['img_local_path']
			img = preprocess(Image.open(path))
			imgs.append(img)

			cap_1, cap_2, y = get_pair_cap(idx, length, labels, i)

			emb_1 = torch.Tensor(model_bert.encode(cap_1))
			emb_2 = torch.Tensor(model_bert.encode(cap_2))

			target.append(y)

			sent_1.append(emb_1)
			sent_2.append(emb_2)

		target = torch.Tensor(target).type(torch.LongTensor).to(torch.device("cuda:0"))

		features, emb_batch_1, emb_batch_2 = get_emb_batch(model_clip, imgs, sent_1, sent_2)


		with torch.no_grad():
			theta = neural(features, emb_batch_1.to(torch.device("cuda:0")), emb_batch_2.to(torch.device("cuda:0")))

		loss += criterion(theta, target)

	neural.train()
	return loss/iters