import sys
sys.path.append('Detection')

# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from efficientnet_pytorch import EfficientNet
from sentence_transformers import SentenceTransformer, util

from torchvision import transforms
from torch import nn
from torch import optim
from PIL import Image
from simple_tools import *
from neurnet import *

import numpy as np
import torch
import json
import os

# config_file = 'Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
# checkpoint_file = '../Data/mask_rcnn_swin_tiny_patch4_window7.pth'

batch_size = 8
epoch = global_step = 0
pre_val = 1

os.makedirs("Model/", exist_ok=True)

transform = transforms.Compose([transforms.Resize((600,600)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

# model_det = init_detector(config_file, checkpoint_file, device='cuda:0')
model_ef = EfficientNet.from_pretrained('efficientnet-b7').to(torch.device("cuda:0"))
model_bert = SentenceTransformer('stsb-mpnet-base-v2')

neural = Neural_Net_Cosine(2560, 768, 256).to(torch.device("cuda:0"))

optimizer = optim.Adam(neural.parameters(), lr=1e-3, weight_decay=4e-5)
cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.3)

f_train = open('../Data/mmsys_anns/train_data.json')
labels = []

f_val = open('../Data/mmsys_anns/val_data.json')
validation = []

for line in f_train:
	labels.append(json.loads(line))

for line in f_val:
	validation.append(json.loads(line))

length = len(labels)
iters = int(length/batch_size)

model_ef.eval()
neural.train()

while epoch < 500:

	# random.shuffle(labels)

	for k in range(iters):

		imgs = []
		sent_1 = []
		sent_2 = []
		y = []

		for i in range(batch_size):

			idx = batch_size*k + i
			
			path = '../Data/' +  labels[i]['img_local_path']
			print(path)
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

		optimizer.zero_grad()

		x_1 = neural(features, emb_batch_1.to(torch.device("cuda:0")))
		x_2 = neural(features, emb_batch_2.to(torch.device("cuda:0")))

		loss = cosine_loss(x_1, x_2, y)
		loss.backward()
		optimizer.step()

		global_step += 1

		if global_step % 100 == 0:
			print("Epoch: %d === Global Step: %d === Loss: %.3f" %(epoch, global_step, loss))

		if global_step % 1000 == 0:
			print("Validation.......")
			val = validation_loss(model_bert, model_ef, neural, validation, cosine_loss)

			if val < pre_val:

				torch.save(neural.state_dict(), 'Model/model_{}.pth'.format(epoch))
				torch.save(optimizer.state_dict(), 'Model/optimizer_{}.pth'.format(epoch))

				txt = open('Model/stat_{}.txt'.format(epoch), 'w')
				txt.write('Loss: %.3f \n'%(loss))
				txt.write('Validation: %.3f'%(val))
				txt.close()

				pre_val = val

			print("Epoch: %d === Global Step: %d === Validation Loss: %.3f" %(epoch, global_step, val))

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