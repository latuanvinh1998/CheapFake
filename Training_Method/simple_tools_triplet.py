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


def get_pair_cap(idx, length, labels):

	cap_1 = labels[idx]["articles"][0]['caption_modified'].replace('\n','')

	if len(labels[idx]["articles"]) > 1:
		for j in range(len(labels[idx]["articles"])):
			if j != len(labels[idx]["articles"]) - 1:
				cap_2 = labels[idx]["articles"][j + 1]['caption_modified'].replace('\n','')
				if difflib.SequenceMatcher(None, cap_1, cap_2).ratio() < 0.8:
					r = random.choice([k for k in range(0,length) if k not in [idx]])
					cap_3 = labels[r]["articles"][random.randint(0, len(labels[r]["articles"]) - 1)]['caption_modified'].replace('\n','')
					return cap_1, cap_2, cap_3

			else:
				return "", "", ""

	else:
		return "", "", ""


def get_emb_batch(model_ef, imgs, sent_1, sent_2, sent_3):

	batch_img = torch.stack([img for img in imgs])

	features = extract(model_ef, batch_img)

	emb_1_batch = torch.stack([emb for emb in sent_1])
	emb_2_batch = torch.stack([emb for emb in sent_2])
	emb_3_batch = torch.stack([emb for emb in sent_3])

	return features, emb_1_batch, emb_2_batch, emb_3_batch


def validation_loss(model_bert, model_ef, neural, labels, loss_function):

	# length = len(labels)

	transform = transforms.Compose([transforms.Resize((600,600)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

	random.shuffle(labels)
	length = 4800
	iters = int(length/8)
	loss = idx = sum_ = 0

	neural.eval()

	while idx < length:

		imgs = []
		sent_1 = []
		sent_2 = []
		sent_3 = []

		batch_num = 0

		while batch_num < 8 and idx < length:

			idx += 1

			cap_1, cap_2, cap_3 = get_pair_cap(idx, length, labels)

			if cap_1 != "":

				path = '../Data/' +  labels[idx]['img_local_path']
				img = transform(Image.open(path))
				imgs.append(img)

				emb_1 = torch.Tensor(model_bert.encode(cap_1))
				emb_2 = torch.Tensor(model_bert.encode(cap_2))
				emb_3 = torch.Tensor(model_bert.encode(cap_3))

				sent_1.append(emb_1)
				sent_2.append(emb_2)
				sent_3.append(emb_3)

				batch_num += 1

		if len(sent_1) != 8:
			continue

		features, emb_batch_1, emb_batch_2, emb_batch_3 = get_emb_batch(model_ef, imgs, sent_1, sent_2, sent_3)

		with torch.no_grad():
			anchor = neural(features, emb_batch_1.to(torch.device("cuda:0")))
			positive = neural(features, emb_batch_2.to(torch.device("cuda:0")))
			negative = neural(features, emb_batch_3.to(torch.device("cuda:0")))

		sum_ += 1
		loss += loss_function(anchor, positive, negative)

	neural.train()
	return loss/sum_