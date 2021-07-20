from torch import nn
from sentence_transformers import SentenceTransformer, util
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from neurnet import *
from simple_tools import *
from torch import optim
from PIL import Image
from scipy import spatial

import json
import torch
import time

f = open('../Data/mmsys_anns/public_test_mmsys_final.json')
labels = []

# f_val = open('../Data/mmsys_anns/val_data.json')
# validation = []

for line in f:
  labels.append(json.loads(line))

# for line in f_val:
#   validation.append(json.loads(line))

length = len(labels)

transform = transforms.Compose([transforms.Resize((600,600)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model_ef = EfficientNet.from_pretrained('efficientnet-b7').to(torch.device("cuda:0"))
model_bert = SentenceTransformer('paraphrase-mpnet-base-v2').to(torch.device("cuda:0"))

neural = Neural_Net_Triplet(2560, 768, 256).to(torch.device("cuda:0"))
neural.load_state_dict(torch.load("Model/model_1.pth"))


TP = TN = FP = FN = idx = total_cos_neg = total_cos_pos = pos = neg =0

label_context = []

thresholds = np.arange(0, 10, 0.05)
cos_thresholds = np.arange(0, 1, 0.05)

for label in labels:

  start = time.time()

  emb_sent_1 = model_bert.encode(label['caption1_modified'])
  emb_sent_2 = model_bert.encode(label['caption2_modified'])
  cos_sim = spatial.distance.cosine(emb_sent_1, emb_sent_2)

  emb_sent_1 = torch.unsqueeze(torch.Tensor(emb_sent_1), 0)
  emb_sent_2 = torch.unsqueeze(torch.Tensor(emb_sent_2), 0)

  path = '../Data/' +  label['img_local_path']
  path = path.replace(':', '_')
  img = transform(Image.open(path)).unsqueeze(0)

  with torch.no_grad():
    feature = model_ef.extract_features(img.to(torch.device("cuda:0")))
    feature = nn.AdaptiveAvgPool2d(1)(feature)
    feature = torch.squeeze(feature, -1)
    feature = torch.squeeze(feature, -1)
    print(time.time() - start)
    raise Exception

    x_1 = neural(feature, emb_sent_1.to(torch.device("cuda:0")))
    x_2 = neural(feature, emb_sent_2.to(torch.device("cuda:0")))

  dis = torch.dist(x_1, x_2)


  label_context.append([label['context_label'], dis, cos_sim])


accs = []
f1s = []
mccs = []

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
    acc, f1, mcc = E1(TP, TN, FP, FN)
    accs.append(acc)
    f1s.append(f1)
    mccs.append(mcc)
print("Accuracy: ",accs[np.argmax(np.asarray(accs))])
print("F1 Score: ", f1s[np.argmax(np.asarray(accs))])
print("MCC: ", mccs[np.argmax(np.asarray(accs))])

print(accs)