from torch import nn
from sentence_transformers import SentenceTransformer, util
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

from neurnet import *
import pandas as pd
import json
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import random


f = open('../Data/mmsys_anns/val_data.json')
labels = []

for line in f:
	labels.append(json.loads(line))

length = len(labels)

transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

model_bert = SentenceTransformer('stsb-mpnet-base-v2').to(torch.device("cuda:0"))
nn_sentence = Neural_Net(768, 256).to(torch.device("cuda:0"))

random.shuffle(labels)
imgs = []
positive = []
negative = []

for i in range(8):

	# print(labels[i]['img_local_path'])
	r = random.choice([k for k in range(0,length) if k not in [i]])
	cap_pos = labels[i]["articles"][random.randint(0, len(labels[i]["articles"]) - 1)]['caption_modified']
	cap_neg = labels[r]["articles"][random.randint(0, len(labels[r]["articles"]) - 1)]['caption_modified']

	pos_emb = torch.Tensor(model_bert.encode(cap_pos))
	neg_emb = torch.Tensor(model_bert.encode(cap_neg))

	positive.append(pos_emb)
	negative.append(neg_emb)

positive_batch = torch.stack([emb for emb in positive])
print(nn_sentence(positive_batch.to(torch.device("cuda:0"))).shape)