from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms
from torch import nn
from PIL import Image
from neurnet import *

import torch
import numpy as np

model = EfficientNet.from_pretrained('efficientnet-b3').to(torch.device("cuda:0"))


transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

img = Image.open('person.jpg')

img = transform(img)
img = torch.unsqueeze(img, 0)

model.eval()
with torch.no_grad():
	score = model(img.to(torch.device("cuda:0"))).cpu().numpy()
	print(np.argmax(score))
# 	features = model.extract_features(img.to(torch.device("cuda:0")))

# features = nn.AdaptiveAvgPool2d(1)(features)
# features = torch.reshape(features, (-1, ))

# model_img = Img_Emb(features.shape[0], 256).to(torch.device("cuda:0"))

# emb = model_img(features)
# print(emb)