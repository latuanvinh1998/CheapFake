
from torch import nn
from sentence_transformers import SentenceTransformer, util
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from neurnet import *
from simple_tools import *

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

# transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

# model_ef = EfficientNet.from_pretrained('efficientnet-b3').to(torch.device("cuda:0"))
# model_bert = SentenceTransformer('stsb-mpnet-base-v2').to(torch.device("cuda:0"))
# neural = Neural_Net_Cosine(1536, 768, 512).to(torch.device("cuda:0"))

# # random.shuffle(labels)
# imgs = []
# positive = []
# negative = []

for i in range(8):

	# path = '../Data/' +  labels[i]['img_local_path']
	# img = Image.open(path)
	# img = transform(img)
	# imgs.append(img)

	r_0 = random.randint(0, len(labels[i]["articles"]) - 1)
	cap_1 = labels[i]["articles"][r_0]['caption_modified']

	if len(labels[i]["articles"]) > 1 and random.randint(0, 1) ==0:
		print(len(labels[i]["articles"]))
		r = random.choice([k for k in range(0, len(labels[i]["articles"]) - 1) if k not in [r_0]])
		cap_2 = labels[i]["articles"][r]['caption_modified']

		print("i: ", i, " r: ", r_0, " r_1: ", r)

	else:
		r = random.choice([k for k in range(0,length) if k not in [i]])
		cap_2 = labels[r]["articles"][random.randint(0, len(labels[r]["articles"]) - 1)]['caption_modified']
		print("i: ", i, " r: ", r,  " r_1: ", r)

# 	pos_emb = torch.Tensor(model_bert.encode(cap_pos))
# 	neg_emb = torch.Tensor(model_bert.encode(cap_neg))

# 	positive.append(pos_emb)
# 	negative.append(neg_emb)

# batch_img = torch.stack([img for img in imgs])
# positive_batch = torch.stack([emb for emb in positive])
# negative_batch = torch.stack([emb for emb in negative])

# feature = extract(model_ef, batch_img)

# x1 = neural(feature, positive_batch.to(torch.device("cuda:0")))
# x2 = neural(feature, negative_batch.to(torch.device("cuda:0")))

# positive_batch = torch.stack([emb for emb in positive])
# print(nn_sentence(positive_batch.to(torch.device("cuda:0"))).shape)

# x1 = torch.Tensor([[-2.3202,  0.9460, -0.8085, -0.7788],
#         [ 0.4801, -0.7071,  3.2686, -0.1520],
#         [ 0.4675, -0.9504,  1.3799,  0.1181]])

# x2 = torch.Tensor([[ 0.7834,  0.8960,  0.0652, -0.4384],
#         [-0.7813, -0.2848, -0.9736,  0.8276],
#         [-1.1003, -1.3273,  1.0929,  1.4579]])

# lb = torch.ones(3)
# y = torch.Tensor([1, 1, -1])
# label = torch.stack([i for i in y])

# l = torch.nn.CosineEmbeddingLoss()

# print(l(x1, x2, label))
# print(l(x1, x2, lb))