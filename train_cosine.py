import sys
sys.path.append('Detection')

# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from efficientnet_pytorch import EfficientNet
from sentence_transformers import SentenceTransformer, util

from torchvision import transforms
from torch import nn
from PIL import Image
from simple_tools import *
from neurnet import *

import numpy as np
import torch
import random
import json

# config_file = 'Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
# checkpoint_file = '../Data/mask_rcnn_swin_tiny_patch4_window7.pth'

batch_size = 8
epoch = global_step = 0

transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

# model_det = init_detector(config_file, checkpoint_file, device='cuda:0')
model_ef = EfficientNet.from_pretrained('efficientnet-b3').to(torch.device("cuda:0"))
model_bert = SentenceTransformer('stsb-mpnet-base-v2')

neural = Neural_Net_Cosine(1536, 768, 512).to(torch.device("cuda:0"))

cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.2)

f = open('../Data/mmsys_anns/val_data.json')
labels = []

for line in f:
	labels.append(json.loads(line))

# random.shuffle(labels)
length = len(labels)
iters = int(length/batch_size)

model_ef.eval()

while epoch < 1:
	for k in range(8):

		imgs = []
		positive = []
		negative = []

		y = -torch.ones(8).to(torch.device("cuda:0"))

		for i in range(batch_size):

			idx = 8*k + i

			path = '../Data/' +  labels[idx]['img_local_path']
			img = Image.open(path)
			img = transform(img)
			imgs.append(img)

			r = random.choice([j for j in range(0,length) if j not in [idx]])
			cap_pos = labels[idx]["articles"][random.randint(0, len(labels[idx]["articles"]) - 1)]['caption_modified']
			cap_neg = labels[r]["articles"][random.randint(0, len(labels[r]["articles"]) - 1)]['caption_modified']

			pos_emb = torch.Tensor(model_bert.encode(cap_pos))
			neg_emb = torch.Tensor(model_bert.encode(cap_neg))

			positive.append(pos_emb)
			negative.append(neg_emb)

		batch_img = torch.stack([img for img in imgs])
		positive_batch = torch.stack([emb for emb in positive])
		negative_batch = torch.stack([emb for emb in negative])

		feature = extract(model_ef, batch_img)


		x_1 = neural(feature, positive_batch.to(torch.device("cuda:0")))
		x_2 = neural(feature, negative_batch.to(torch.device("cuda:0")))

		loss = cosine_loss(x_1, x_2, y)
		print(loss)

	epoch += 1

print("Ok!~")


# img_emb = []

# result = inference_detector(model_det, path)
# with torch.no_grad():
# 	for i in range(80):
# 		for r in result[0][i]:
# 			if r[4] > 0.8:
# 				bb = r[0:4].astype(int)

# 				img_crop = img.crop((bb[0], bb[1], bb[2], bb[3]))
# 				img_crop = transform(img_crop)
# 				img_crop = torch.unsqueeze(img_crop, 0)

# 				features = model_ef.extract_features(img_crop.to(torch.device("cuda:0")))
# 				features = nn.AdaptiveAvgPool2d(1)(features)
# 				features = torch.flatten(features)
# 				img_emb.append(features)

# img_vec = []

# for i in range(len(img_emb)):
# 	emb_ = torch.reshape(img_emb[i], (1, -1))
# 	emb = nn_image(img_emb[i])
# 	print(emb.shape)
# 	img_vec.append(emb)

# select_triplets(img_vec, emb_sent_1, emb_sent_2)