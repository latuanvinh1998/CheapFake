import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from PIL import Image

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from efficientnet_pytorch import EfficientNet
from sentence_transformers import SentenceTransformer, util
from simple_tools import *
from neurnet import *

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

os.makedirs("Model/", exist_ok=True)
os.makedirs("Img/", exist_ok=True)

transform = transforms.Compose([transforms.Resize((380,380)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

model_ef = EfficientNet.from_pretrained('efficientnet-b4').to(torch.device("cuda:0"))
model_bert = SentenceTransformer('paraphrase-mpnet-base-v2')
model = BaseLine(1792, 768, 256).to(torch.device("cuda:0"))

batch_size = 8
epoch = global_step = 0
pre_val = 1

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

while epoch < 1:

	for k in range(1):
		img_1 = []
		img_2 = []
		sent_1 = []
		sent_2 = []
		y = []

		for i in range(1):
			idx = batch_size*k + i
			
			path = '../Data/' +  labels[i]['img_local_path']
			len_bb = get_bb(path, predictor)

			cap_1, cap_2 = get_pair(idx, length, labels)

			emb_1 = torch.Tensor(model_bert.encode(cap_1)).to(torch.device("cuda:0")).unsqueeze(0)
			emb_2 = torch.Tensor(model_bert.encode(cap_2)).to(torch.device("cuda:0")).unsqueeze(0)

			print(emb_1.shape)

			img = transform(Image.open('Img/' + str(0) + '.jpg')).unsqueeze(0).to(torch.device("cuda:0"))

			with torch.no_grad():
				emb_img = model_ef(img)
				emb = model(emb_img, emb_1)
			print(emb)
			print(emb.shape)

		# 	img_1.append(choose_bb(emb_1, model_ef, transform, len_bb))
		# 	img_2.append(choose_bb(emb_2, model_ef, transform, len_bb))

		# 	sent_1.append(emb_1)
		# 	sent_2.append(emb_2)

		# batch_img_1 = torch.stack([img for img in img_1])
		# batch_img_2 = torch.stack([img for img in img_2])

		# batch_sent_1 = torch.stack([emb for emb in sent_1])
		# batch_sent_2 = torch.stack([emb for emb in sent_2])

	epoch += 1
