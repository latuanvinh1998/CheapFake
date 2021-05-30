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


def get_pair_cap(idx, length, labels, i):

	cap_1 = labels[idx]["articles"][0]['caption_modified'].replace('\n','')

	if len(labels[idx]["articles"]) > 1 and i <= 5:
		for j in range(len(labels[idx]["articles"])):
			if j != len(labels[idx]["articles"]) - 1:
				cap_2 = labels[idx]["articles"][j + 1]['caption_modified'].replace('\n','')
				if difflib.SequenceMatcher(None, cap_1, cap_2).ratio() < 0.5:
					return cap_1, cap_2, 1

			else:
				r = random.choice([k for k in range(0,length) if k not in [idx]])
				cap_2 = labels[r]["articles"][random.randint(0, len(labels[r]["articles"]) - 1)]['caption_modified'].replace('\n','')
				return cap_1, cap_2, -1


	else:
		r = random.choice([k for k in range(0,length) if k not in [idx]])
		cap_2 = labels[r]["articles"][random.randint(0, len(labels[r]["articles"]) - 1)]['caption_modified'].replace('\n','')
		return cap_1, cap_2, -1


def get_emb_batch(model_ef, imgs, sent_1, sent_2):

	batch_img = torch.stack([img for img in imgs])

	features = extract(model_ef, batch_img)

	emb_1_batch = torch.stack([emb for emb in sent_1])
	emb_2_batch = torch.stack([emb for emb in sent_2])

	return features, emb_1_batch, emb_2_batch


def validation_loss(model_bert, model_ef, neural, labels, cosine_loss):

	# length = len(labels)
	random.shuffle(labels)
	length = 1600
	iters = int(length/8)
	loss = 0

	transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

	neural.eval()

	for k in range(iters):

		imgs = []
		sent_1 = []
		sent_2 = []
		y = []

		for i in range(8):

			idx = 8*k + i
			
			path = '../Data/' +  labels[i]['img_local_path']
			img = Image.open(path)
			img = transform(img)
			imgs.append(img)

			cap_1, cap_2, y_ = get_pair_cap(idx, length, labels, i)
			y.append(y_)

			emb_1 = torch.Tensor(model_bert.encode(cap_1))
			emb_2 = torch.Tensor(model_bert.encode(cap_2))

			sent_1.append(emb_1)
			sent_2.append(emb_2)

		features, emb_batch_1, emb_batch_2 = get_emb_batch(model_ef, imgs, sent_1, sent_2)

		y = torch.Tensor(y).to(torch.device("cuda:0"))

		with torch.no_grad():
			x_1 = neural(features, emb_batch_1.to(torch.device("cuda:0")))
			x_2 = neural(features, emb_batch_2.to(torch.device("cuda:0")))

		loss += cosine_loss(x_1, x_2, y).detach().cpu().numpy()

	neural.train()
	return loss/iters