import sys
sys.path.append('Detection')

# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import clip
from sentence_transformers import SentenceTransformer, util

from torchvision import transforms
from torch import nn
from torch import optim
from PIL import Image
from simple_tools_classify import *
from neurnet import *

import numpy as np
import torch
import json
import os

# config_file = 'Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
# checkpoint_file = '../Data/mask_rcnn_swin_tiny_patch4_window7.pth'

batch_size = 8
epoch = 1
global_step = Accumulate_Loss = 0
pre_val = 10

os.makedirs("Model/", exist_ok=True)


# model_det = init_detector(config_file, checkpoint_file, device='cuda:0')
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)
model_bert = SentenceTransformer('paraphrase-mpnet-base-v2')

neural = Neural_Net(512, 768).to(torch.device("cuda:0"))

optimizer = optim.Adam(neural.parameters(), lr=1e-4, weight_decay=4e-5)
criterion = nn.CrossEntropyLoss()

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

model_clip.eval()
neural.train()

while epoch < 500:

	random.shuffle(labels)

	for k in range(iters):

		imgs = []
		sent_1 = []
		sent_2 = []
		target = []

		for i in range(batch_size):

			idx = batch_size*k + i
			
			path = '../Data/' +  labels[idx]['img_local_path']
			img = preprocess(Image.open(path))
			imgs.append(img)

			cap_1, cap_2, y = get_pair_cap(idx, length, labels, i)

			emb_1 = torch.Tensor(model_bert.encode(cap_1))
			emb_2 = torch.Tensor(model_bert.encode(cap_2))
			target.append(y)

			sent_1.append(emb_1)
			sent_2.append(emb_2)

		target = torch.Tensor(target).type(torch.LongTensor).to(torch.device("cuda:0"))

		features, emb_batch_1, emb_batch_2 = get_emb_batch(model_clip, imgs, sent_1, sent_2)

		optimizer.zero_grad()

		theta = neural(features, emb_batch_1.to(torch.device("cuda:0")), emb_batch_2.to(torch.device("cuda:0")))

		loss = criterion(theta, target)
		loss.backward()
		optimizer.step()

		global_step += 1

		Accumulate_Loss += loss.item()

		if global_step % 100 == 0:
			a_loss = Accumulate_Loss/100
			print("Epoch: %d === Global Step: %d === Loss: %.3f" %(epoch, global_step, a_loss))
			Accumulate_Loss = 0

		if global_step % 1000 == 0:
			print("Validation.......")
			val = validation_loss(model_bert, model_clip, neural, validation, criterion, preprocess)

			if val < pre_val:

				torch.save(neural.state_dict(), 'Model/model_{}.pth'.format(epoch))
				torch.save(optimizer.state_dict(), 'Model/optimizer_{}.pth'.format(epoch))

				txt = open('Model/stat_{}.txt'.format(epoch), 'w')
				txt.write('Loss: %.3f \n'%(a_loss))
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