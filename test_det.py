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

config_file = 'Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
checkpoint_file = '../Data/mask_rcnn_swin_tiny_patch4_window7.pth'

transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

model_det = init_detector(config_file, checkpoint_file, device='cuda:0')
model_ef = EfficientNet.from_pretrained('efficientnet-b3').to(torch.device("cuda:0"))
model_nlp = SentenceTransformer('stsb-mpnet-base-v2')
nn_image = Img_Emb(1536, 256)
nn_sentence = Caption_Emb(768, 256)

result = inference_detector(model_det, 'test.jpg')

img = Image.open('test.jpg')
img_emb = []

model_ef.eval()

with torch.no_grad():
	for r in result[0][2]:
		if r[4] > 0.9:
			bb = r[0:4].astype(int)

			img_crop = img.crop((bb[0], bb[1], bb[2], bb[3]))
			img_crop.show()
			img_crop = transform(img_crop)
			img_crop = torch.unsqueeze(img_crop, 0)

			features = model_ef.extract_features(img_crop.to(torch.device("cuda:0")))
			features = nn.AdaptiveAvgPool2d(1)(features)
			features = torch.flatten(features)
			img_emb.append(features)

sent_emb_1 = torch.flatten(torch.tensor(model_nlp.encode("Hello Darkness My Friend")))
sent_emb_1 = nn_sentence(sent_emb_1)

# sent_emb_2 = torch.tensor(model_nlp.encode("I'm facking Gay"))
