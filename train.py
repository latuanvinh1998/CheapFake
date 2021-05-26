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

# config_file = 'Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
# checkpoint_file = '../Data/mask_rcnn_swin_tiny_patch4_window7.pth'

# transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

# model_det = init_detector(config_file, checkpoint_file, device='cuda:0')
# model_ef = EfficientNet.from_pretrained('efficientnet-b3').to(torch.device("cuda:0"))
# model_nlp = SentenceTransformer('stsb-mpnet-base-v2')

# nn_image = Neural_Net(1536, 256)
# nn_sentence = Neural_Net(768, 256)

f = open('../Data/mmsys_anns/public_test_mmsys_final.json')
labels = []

for line in f:
	labels.append(json.loads(line))

path = '../Data/' +  labels[0]['img_local_path']

cap_1 = labels[0]['caption1_modified']
cap_2 = labels[0]['caption2_modified']

