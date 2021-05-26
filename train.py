import sys
sys.path.append('Detection')

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
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

config_file = 'Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
checkpoint_file = '../Data/mask_rcnn_swin_tiny_patch4_window7.pth'

transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

model_det = init_detector(config_file, checkpoint_file, device='cuda:0')
model_ef = EfficientNet.from_pretrained('efficientnet-b3').to(torch.device("cuda:0"))
model_bert = SentenceTransformer('stsb-mpnet-base-v2').to(torch.device("cuda:0"))

nn_image = Neural_Net(1536, 256).to(torch.device("cuda:0"))
nn_sentence = Neural_Net(768, 256).to(torch.device("cuda:0"))

f = open('../Data/mmsys_anns/public_test_mmsys_final.json')
labels = []

for line in f:
	labels.append(json.loads(line))

path = '../Data/' +  labels[0]['img_local_path']

cap_1 = labels[0]['caption1_modified']
cap_2 = labels[0]['caption2_modified']

emb_sent_1 = torch.Tensor(model_bert.encode(cap_1))
emb_sent_2 = torch.Tensor(model_bert.encode(cap_2))

emb_sent_1 = torch.reshape(emb_sent_1,(1, -1))
emb_sent_2 = torch.reshape(emb_sent_2,(1, -1))

emb_sent_1 = nn_sentence(emb_sent_1.to(torch.device("cuda:0")))
emb_sent_2 = nn_sentence(emb_sent_2.to(torch.device("cuda:0")))

img = Image.open(path)
img_emb = []

result = inference_detector(model_det, path)
with torch.no_grad():
	for i in range(80):
		for r in result[0][i]:
			if r[4] > 0.8:
				bb = r[0:4].astype(int)

				img_crop = img.crop((bb[0], bb[1], bb[2], bb[3]))
				img_crop = transform(img_crop)
				img_crop = torch.unsqueeze(img_crop, 0)

				features = model_ef.extract_features(img_crop.to(torch.device("cuda:0")))
				features = nn.AdaptiveAvgPool2d(1)(features)
				features = torch.flatten(features)
				img_emb.append(features)

img_vec = []

for i in range(len(img_emb)):
	emb_ = torch.reshape(img_emb[i], (1, -1))
	emb = nn_image(img_emb[i])
	print(emb.shape)
	img_vec.append(emb)

select_triplets(img_vec, emb_sent_1, emb_sent_2)