import sys
sys.path.append('Segment')

from sentence_transformers import SentenceTransformer, util
from scipy import spatial
from simple_tools import *
from mmseg.apis import inference_segmentor, init_segmentor

import numpy as np
import pandas as pd
import json

f = open('../Data/mmsys_anns/public_test_mmsys_final.json')
label_csv = pd.read_csv("objectInfo150.csv")

labels = []
for line in f:
	labels.append(json.loads(line))

TP = TN = FP = FN = 0

config_file = 'Segment/configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py'
checkpoint_file = '../Data/upernet_swin_small_patch4_window7_512x512.pth'

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

model_bert = SentenceTransformer('stsb-mpnet-base-v2')

for label in labels:
	emb_sent_1 = model_bert.encode(label['caption1_modified'])
	emb_sent_2 = model_bert.encode(label['caption2_modified'])
	cos_sim = spatial.distance.cosine(emb_sent_1, emb_sent_2)

	if cos_sim < 0.5:
		pred = 0
		path = '../Data/' +  label['img_local_path']
		result = inference_segmentor(model, path.replace(':', '_'))

		arr = np.unique(result[0])
		score_1 = np.copy(arr)
		score_2 = np.copy(arr)

		for i in range(len(arr)):
			emb_label = model_bert.encode(label_csv['Name'][arr[i]].split(';')[0])

			cos_img_1 = spatial.distance.cosine(emb_sent_1, emb_label)
			cos_img_2 = spatial.distance.cosine(emb_sent_2, emb_label)
			score_1[i] = cos_img_1
			score_2[i] = cos_img_2

		if abs(score_1[np.argmin(score_1)] - score_2[np.argmin(score_1)]) < 0.05 or abs(score_1[np.argmin(score_2)] - score_2[np.argmin(score_2)]) < 0.05:
			pred = 0
		else:
			pred = 1

	else:
		pred = 1
	if label['context_label'] == 0 and pred == 0:
		TN += 1
	elif label['context_label'] == 1 and pred == 1:
		TP += 1
	elif label['context_label'] == 1 and pred == 0:
		FN += 1
	elif label['context_label'] == 0 and pred == 1:
		FP += 1
acc, f1, mcc = E1(TP, TN, FP, FN)
print('Accuracy: ', acc*100, '%')
print('f1: ', f1)
print('mcc: ', mcc)