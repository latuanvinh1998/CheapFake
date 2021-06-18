
from torch import nn
from sentence_transformers import SentenceTransformer, util
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from neurnet import *
from simple_tools import *
from torch import optim
from PIL import Image
from scipy import spatial

import pandas as pd
import json
import torch

f = open('../Data/mmsys_anns/public_test_mmsys_final.json')
labels = []

# f_val = open('../Data/mmsys_anns/val_data.json')
# validation = []

for line in f:
	labels.append(json.loads(line))

# for line in f_val:
# 	validation.append(json.loads(line))

length = len(labels)

transform = transforms.Compose([transforms.Resize((600,600)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

model_ef = EfficientNet.from_pretrained('efficientnet-b7').to(torch.device("cuda:0"))
model_bert = SentenceTransformer('stsb-mpnet-base-v2').to(torch.device("cuda:0"))

neural = Neural_Net_Cosine(2560, 768, 256).to(torch.device("cuda:0"))
neural.load_state_dict(torch.load("Model/model_1.pth"))

TP = TN = FP = FN = idx = total_cos_neg = total_cos_pos = pos = neg =0

label_context = []

thresholds = np.arange(0, 1, 0.01)
cos_thresholds = np.arange(0, 1, 0.05)

for label in labels:

	emb_sent_1 = model_bert.encode(label['caption1_modified'])
	emb_sent_2 = model_bert.encode(label['caption2_modified'])
	cos_sim = spatial.distance.cosine(emb_sent_1, emb_sent_2)

	emb_sent_1 = torch.unsqueeze(torch.Tensor(emb_sent_1), 0)
	emb_sent_2 = torch.unsqueeze(torch.Tensor(emb_sent_2), 0)

	path = '../Data/' +  label['img_local_path']
	path = path.replace(':', '_')
	img = Image.open(path)
	img = transform(img)
	img = torch.unsqueeze(img, 0)

	with torch.no_grad():
		feature = model_ef.extract_features(img.to(torch.device("cuda:0")))
		feature = nn.AdaptiveAvgPool2d(1)(feature)
		feature = torch.squeeze(feature, -1)
		feature = torch.squeeze(feature, -1)

		x_1 = neural(feature, emb_sent_1.to(torch.device("cuda:0")))
		x_2 = neural(feature, emb_sent_2.to(torch.device("cuda:0")))

	cos = spatial.distance.cosine(x_1.cpu().numpy(), x_2.cpu().numpy())
	label_context.append([label['context_label'], cos, cos_sim])


accs = []

for threshold in thresholds:
	TP = TN = FP = FN = 0

	for label, cos, cos_sim in label_context:

		if cos < threshold and label == 0:
			TN += 1
		elif cos > threshold and label == 0:
			FP += 1
		elif cos > threshold and label == 1:
			TP += 1
		elif cos < threshold and label == 1:
			FN += 1

	if TP != 0 and TN != 0 and FP != 0 and FN != 0:
		acc, f1, mcc = E1(TP, TN, FP, FN)
		accs.append(acc)
print("Accuracy: ",accs[np.argmax(np.asarray(accs))])
# if accs != []:
# 	print("Cos threshold: ",accs[np.argmax(np.asarray(accs))])
# for cos_threshold in cos_thresholds:

# 	accs = []

# 	for threshold in thresholds:
# 		TP = TN = FP = FN = 0

# 		for label, cos, cos_sim in label_context:

# 			if cos_sim < cos_threshold:
# 				if cos < threshold and label == 0:
# 					TN += 1
# 				elif cos > threshold and label == 0:
# 					FP += 1
# 				elif cos > threshold and label == 1:
# 					TP += 1
# 				elif cos < threshold and label == 1:
# 					FN += 1
# 			else:
# 				if label == 0:
# 					FP += 1
# 				else:
# 					TP += 1

# 		if TP != 0 and TN != 0 and FP != 0 and FN != 0:
# 			acc, f1, mcc = E1(TP, TN, FP, FN)
# 			accs.append(acc)
# 	if accs != []:
# 		print("Cos threshold: ",cos_threshold, "\t", accs[np.argmax(np.asarray(accs))])
# print(accs[np.argmax(np.array(accs))])

# for acc in accs:
# 	print(acc)

	# if cos < 0.35 and label['context_label'] == 1:
	# 	neg += 1
	# elif cos > 0.4 and label['context_label'] == 0:
	# 	pos += 1

	# else:
	# 	pred = 1

# print("Not Out of context: ", total_cos_neg/neg)
# print("Out of context: ", total_cos_pos/pos)
	# if label['context_label'] == 0 and pred == 0:
	# 	TN += 1
	# elif label['context_label'] == 1 and pred == 1:
	# 	TP += 1
	# elif label['context_label'] == 1 and pred == 0:
	# 	FN += 1
	# elif label['context_label'] == 0 and pred == 1:
	# 	FP += 1

# acc, f1, mcc = E1(TP, TN, FP, FN)
# print('Accuracy: ', acc*100, '%')