import sys
sys.path.append('Detection')

# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from efficientnet_pytorch import EfficientNet
from sentence_transformers import SentenceTransformer, util

from torchvision import transforms
from torch import nn
from PIL import Image
from simple_tools_triplet import *
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
pre_val = 1

os.makedirs("Model/", exist_ok=True)

transform = transforms.Compose([transforms.Resize((600,600)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

# model_det = init_detector(config_file, checkpoint_file, device='cuda:0')
model_ef = EfficientNet.from_pretrained('efficientnet-b7').to(torch.device("cuda:0"))
model_bert = SentenceTransformer('stsb-mpnet-base-v2')

neural = Neural_Net_Triplet(2560, 768, 256).to(torch.device("cuda:0"))

optimizer = torch.optim.Adam(neural.parameters(), lr=1e-3, weight_decay=4e-5)
triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2)

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

neural.load_state_dict(torch.load('Model/model_1.pth'))
optimizer.load_state_dict(torch.load('Model/optimizer_1.pth'))

model_ef.eval()

neural.train()

while epoch < 500:

	random.shuffle(labels)
	idx = 0

	while idx < length - 1:

		imgs = []
		sent_1 = []
		sent_2 = []
		sent_3 = []

		batch_num = 0

		while batch_num < batch_size and idx < length - 1:

			idx += 1

			cap_1, cap_2, cap_3 = get_pair_cap(idx, length, labels)

			if cap_1 != "":

				path = '../Data/' +  labels[idx]['img_local_path']
				img = transform(Image.open(path))
				imgs.append(img)

				emb_1 = torch.Tensor(model_bert.encode(cap_1))
				emb_2 = torch.Tensor(model_bert.encode(cap_2))
				emb_3 = torch.Tensor(model_bert.encode(cap_3))

				sent_1.append(emb_1)
				sent_2.append(emb_2)
				sent_3.append(emb_3)

				batch_num += 1

		if len(sent_1) != 8:
			continue

		features, emb_batch_1, emb_batch_2, emb_batch_3 = get_emb_batch(model_ef, imgs, sent_1, sent_2, sent_3)

		optimizer.zero_grad()

		anchor = neural(features, emb_batch_1.to(torch.device("cuda:0")))
		positive = neural(features, emb_batch_2.to(torch.device("cuda:0")))
		negative = neural(features, emb_batch_3.to(torch.device("cuda:0")))

		loss = triplet_loss(anchor, positive, negative)
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
			val = validation_loss(model_bert, model_ef, neural, validation, triplet_loss)

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
