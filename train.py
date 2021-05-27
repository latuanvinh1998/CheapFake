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
import json

# config_file = 'Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
# checkpoint_file = '../Data/mask_rcnn_swin_tiny_patch4_window7.pth'

batch_size = 16
epoch = global_step = 0

transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

# model_det = init_detector(config_file, checkpoint_file, device='cuda:0')
model_ef = EfficientNet.from_pretrained('efficientnet-b3').to(torch.device("cuda:0"))
model_bert = SentenceTransformer('stsb-mpnet-base-v2').to(torch.device("cuda:0"))

nn_image = Neural_Net(1536, 256).to(torch.device("cuda:0"))
nn_sentence = Neural_Net(768, 256).to(torch.device("cuda:0"))

f = open('../Data/mmsys_anns/train_data.json')
labels = []

for line in f:
	labels.append(json.loads(line))

random.shuffle(labels)
length = len(labels)
iters = int(length/batch_size)

cap_1 = labels[0]['caption1_modified']
cap_2 = labels[0]['caption2_modified']

emb_sent_1 = torch.Tensor(model_bert.encode(cap_1))
emb_sent_2 = torch.Tensor(model_bert.encode(cap_2))

emb_sent_1 = torch.reshape(emb_sent_1,(1, -1))
emb_sent_2 = torch.reshape(emb_sent_2,(1, -1))

positive = nn_sentence(emb_sent_1.to(torch.device("cuda:0")))
negative = nn_sentence(emb_sent_2.to(torch.device("cuda:0")))

while epoch < 1000:
	for k in range(iters):

		imgs = []
		positive = []
		negative = []

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

		anchor = nn_image(feature)
		positive_anchor = nn_sentence(positive_batch.to(torch.device("cuda:0")))
		negative_anchor = nn_sentence(negative_batch.to(torch.device("cuda:0")))




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