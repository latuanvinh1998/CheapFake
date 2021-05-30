
from torch import nn
from sentence_transformers import SentenceTransformer, util
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from neurnet import *
from simple_tools import *
from torch import optim
from PIL import Image

import pandas as pd
import json
import torch

f = open('../Data/mmsys_anns/train_data.json')
labels = []

f_val = open('../Data/mmsys_anns/val_data.json')
validation = []

for line in f:
	labels.append(json.loads(line))

for line in f_val:
	validation.append(json.loads(line))

length = len(labels)

transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

model_ef = EfficientNet.from_pretrained('efficientnet-b3').to(torch.device("cuda:0"))
model_bert = SentenceTransformer('stsb-mpnet-base-v2').to(torch.device("cuda:0"))
neural = Neural_Net_Cosine(1536, 768, 512).to(torch.device("cuda:0"))

optimizer = optim.Adam(neural.parameters(), lr=1e-3)
cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.2)

# random.shuffle(labels)

model_ef.eval()
neural.train()

# validation_loss(model_bert, model_ef, neural, validation, cosine_loss)

# for i in range(1):
# 	# path = '../Data/' +  validation[i]['img_local_path']
# 	# print(path)
# 	sent_1 = []
# 	sent_2 = []
# 	y = []
# 	for j in range(8):
# 		idx = 8*i + j
# 		cap_1, cap_2, y_ = get_pair_cap(idx, length, labels, j)
# 		y.append(y_)

# 		print("idx: ", idx)
# 		print(cap_1)
# 		print(cap_2)
# 	y = torch.Tensor(y).to(torch.device("cuda:0"))
# 	print(y)

for i in range(1):

	imgs = []
	sent_1 = []
	sent_2 = []
	y = []
	for l in range(8):

		idx = 8*i + l
		path = '../Data/' +  labels[i]['img_local_path']
		img = Image.open(path)
		img = transform(img)
		imgs.append(img)

		cap_1, cap_2, y_ = get_pair_cap(idx, length, labels, l)
		y.append(y_)

		emb_1 = torch.Tensor(model_bert.encode(cap_1))
		emb_2 = torch.Tensor(model_bert.encode(cap_2))

		sent_1.append(emb_1)
		sent_2.append(emb_2)

	features, emb_batch_1, emb_batch_2 = get_emb_batch(model_ef, imgs, sent_1, sent_2)
	y = torch.Tensor(y).to(torch.device("cuda:0"))

	# optimizer.zero_grad()

	x_1 = neural(features, emb_batch_1.to(torch.device("cuda:0")))
	x_2 = neural(features, emb_batch_2.to(torch.device("cuda:0")))

	print(x_1.shape)

# 	loss = cosine_loss(x_1, x_2, y)
# 	print(loss.detach().cpu().numpy())
	# loss.backward()
	# optimizer.step()
