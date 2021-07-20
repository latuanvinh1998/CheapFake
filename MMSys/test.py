from torch import nn
from sentence_transformers import SentenceTransformer, util
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import clip
from neurnet import *
from simple_tools import *
from torch import optim
from PIL import Image
from scipy import spatial

import json
import time
import torch

f = open('./mmsys21cheapfakes/mmsys_anns/public_test_mmsys_final.json')
labels = []

# f_val = open('../Data/mmsys_anns/val_data.json')
# validation = []

for line in f:
	labels.append(json.loads(line))

# for line in f_val:
	# validation.append(json.loads(line))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

length = len(labels)

transform = transforms.Compose([transforms.Resize((600,600)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])



model_ef = EfficientNet.from_pretrained('efficientnet-b7').to(device)
model_clip, preprocess = clip.load("ViT-B/32", device=device)
model_bert = SentenceTransformer('paraphrase-mpnet-base-v2').to(device)
neural = Neural_Net_Cosine(2560, 768, 256).to(device)
neural.load_state_dict(torch.load("./source_code/Model/model_prelu.pth", map_location ='cpu'))
neural = neural.to(device)


TP = TN = FP = FN = idx = total_cos_neg = total_cos_pos = pos = neg =0

label_context = []

thresholds = np.arange(0, 1, 0.01)
cos_thresholds = np.arange(0, 1, 0.05)
start = time.time()

print("\nCosine Similarity Method result:")

for label in labels:

	emb_sent_1 = model_bert.encode(label['caption1_modified'])
	emb_sent_2 = model_bert.encode(label['caption2_modified'])
	cos_sim = spatial.distance.cosine(emb_sent_1, emb_sent_2)

	emb_sent_1 = torch.unsqueeze(torch.Tensor(emb_sent_1), 0)
	emb_sent_2 = torch.unsqueeze(torch.Tensor(emb_sent_2), 0)

	path = './mmsys21cheapfakes/' + label['img_local_path']
	path = path.replace(':', '_')
	img = transform(Image.open(path)).unsqueeze(0)

	with torch.no_grad():
		feature = model_ef.extract_features(img.to(device))
		feature = nn.AdaptiveAvgPool2d(1)(feature)
		feature = torch.squeeze(feature, -1)
		feature = torch.squeeze(feature, -1)

		x_1 = neural(feature, emb_sent_1.to(device))
		x_2 = neural(feature, emb_sent_2.to(device))
		
	cos = spatial.distance.cosine(x_1.cpu().numpy(), x_2.cpu().numpy())
	label_context.append([label['context_label'], cos, cos_sim])

print("\tLatency: ", (time.time() - start))

accs = []
f1s = []
mccs = []
precisions = []
recalls = []

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
		acc, f1, mcc, precision, recall = E1(TP, TN, FP, FN)
		accs.append(acc)
		f1s.append(f1)
		mccs.append(mcc)
		precisions.append(precision)
		recalls.append(recall)
print("\tAccuracy: ",accs[np.argmax(np.asarray(accs))])
print("\tPrecision: ", precisions[np.argmax(np.asarray(accs))])
print("\tRecall: ", recalls[np.argmax(np.asarray(accs))])
print("\tF1 Score: ", f1s[np.argmax(np.asarray(accs))])
print("\tMCC: ", mccs[np.argmax(np.asarray(accs))])


############## CLASSIFY ######################

print("Classify method result: ")
neural = Neural_Net(512, 768)
neural.load_state_dict(torch.load("./source_code/Model/model_classify.pth", map_location ='cpu'))
neural = neural.to(device)

start = time.time()

for label in labels:

	emb_sent_1 = model_bert.encode(label['caption1_modified'])
	emb_sent_2 = model_bert.encode(label['caption2_modified'])
	cos_sim = spatial.distance.cosine(emb_sent_1, emb_sent_2)

	emb_sent_1 = torch.unsqueeze(torch.Tensor(emb_sent_1), 0)
	emb_sent_2 = torch.unsqueeze(torch.Tensor(emb_sent_2), 0)

	path = './mmsys21cheapfakes/' +  label['img_local_path']
	path = path.replace(':', '_')
	img = preprocess(Image.open(path)).unsqueeze(0)

	with torch.no_grad():
		feature = model_clip.encode_image(img.to(device)).type(torch.FloatTensor)
		feature = feature.to(device)
		theta = neural(feature, emb_sent_1.to(device), emb_sent_2.to(device))

	# print(nn.Softmax(dim=1)(x_2))

	label_ = np.argmax(theta.cpu().numpy())

	if label['context_label'] == 0 and label_ == 0:
		TN += 1
	elif label['context_label'] == 1 and label_ == 1:
		TP += 1
	elif label['context_label'] == 0 and label_ == 1:
		FP += 1
	elif label['context_label'] == 1 and label_ == 0:
		FN += 1

print("\tLatency: ", (time.time() - start))

acc, f1, mcc, precision, recall = E1(TP, TN, FP, FN)

print("\tAccuracy: ",acc)
print("\tPrecision: ",precision)
print("\tRecall: ", recall)
print("\tF1 Score: ", f1)
print("\tMCC: ", mcc)

################ TRIPLET ##############################

print("Euclidean Distance method result: ")

neural = Neural_Net_Triplet(2560, 768, 256)
neural.load_state_dict(torch.load("./source_code/Model/model_triplet.pth", map_location ='cpu'))
neural = neural.to(device)

label_context = []

start = time.time()

for label in labels:

	emb_sent_1 = model_bert.encode(label['caption1_modified'])
	emb_sent_2 = model_bert.encode(label['caption2_modified'])
	cos_sim = spatial.distance.cosine(emb_sent_1, emb_sent_2)

	emb_sent_1 = torch.unsqueeze(torch.Tensor(emb_sent_1), 0)
	emb_sent_2 = torch.unsqueeze(torch.Tensor(emb_sent_2), 0)

	path = './mmsys21cheapfakes/' +  label['img_local_path']
	path = path.replace(':', '_')
	img = transform(Image.open(path)).unsqueeze(0)

	with torch.no_grad():
		feature = model_ef.extract_features(img.to(device))
		feature = nn.AdaptiveAvgPool2d(1)(feature)
		feature = torch.squeeze(feature, -1)
		feature = torch.squeeze(feature, -1)

		x_1 = neural(feature, emb_sent_1.to(device))
		x_2 = neural(feature, emb_sent_2.to(device))

	dis = torch.dist(x_1, x_2)


	label_context.append([label['context_label'], dis, cos_sim])

print("\tLatency: ", (time.time() - start))


accs = []
f1s = []
mccs = []
precisions = []
recalls = []

thresholds = np.arange(0, 10, 0.05)


for threshold in thresholds:
	TP = TN = FP = FN = 0

	for label, dis, cos_sim in label_context:

		if dis < threshold and label == 0:
			TN += 1
		elif dis > threshold and label == 0:
			FP += 1
		elif dis > threshold and label == 1:
			TP += 1
		elif dis < threshold and label == 1:
			FN += 1

	if TP != 0 and TN != 0 and FP != 0 and FN != 0:
		acc, f1, mcc, precision, recall = E1(TP, TN, FP, FN)
		accs.append(acc)
		f1s.append(f1)
		mccs.append(mcc)
		precisions.append(precision)
		recalls.append(recall)
print("\tAccuracy: ",accs[np.argmax(np.asarray(accs))])
print("\tPrecision: ", precisions[np.argmax(np.asarray(accs))])
print("\tRecall: ", recalls[np.argmax(np.asarray(accs))])
print("\tF1 Score: ", f1s[np.argmax(np.asarray(accs))])
print("\tMCC: ", mccs[np.argmax(np.asarray(accs))])