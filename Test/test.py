from torch import nn
from sentence_transformers import SentenceTransformer, util
# from neurnet import *
import pandas as pd
import json
import torch
import cv2

# triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

# anchor = torch.randn(100, 128)
# positive = torch.randn(100, 128)
# negative = torch.randn(100, 128)

# nn_sentence = Caption_Emb(768, 256)

# # output = triplet_loss(anchor, positive, positive)
# # print(output)

# model_nlp = SentenceTransformer('stsb-mpnet-base-v2')
# sent_emb_1 = torch.flatten(torch.tensor(model_nlp.encode("Hello Darkness My Friend")))
# print(nn_sentence(sent_emb_1).shape)

f = open('val.json')
label_csv = pd.read_csv("objectInfo150.csv")

labels = []
for line in f:
	labels.append(json.loads(line))

print(labels[0]["articles"][0]['caption_modified'])

# path = '../Data/' + labels[0]['img_local_path'] 
# img = cv2.imread(path)

# for bb in labels[0]['maskrcnn_bboxes']:
# 	for i in range(4):
# 		bb[i] = int(bb[i])
# 	print(bb)
# 	img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
# cv2.imshow('img', img)
# cv2.waitKey(0)