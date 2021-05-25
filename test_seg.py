import sys
sys.path.append('Segment')
from mmseg.apis import inference_segmentor, init_segmentor
from sentence_transformers import SentenceTransformer, util
from scipy import spatial

import numpy as np
import pandas as pd

label = pd.read_csv("objectInfo150.csv")

config_file = 'Segment/configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py'
checkpoint_file = '../Data/upernet_swin_small_patch4_window7_512x512.pth'

caption_1 = "Supporters of Tanzania's ruling Chama Cha Mapinduzi party come out on Friday to celebrate their candidate's victory in the disputed Zanzibari presidential election"
caption_2 = "A person sits on a truck as supporters of the ruling Chama Cha Mapinduzi (Revolutionary Party) celebrate the victory of their candidate in the Zanzibar Presidential election on the outskirts of Stone Town, on October 30, 2020."

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

model_bert = SentenceTransformer('stsb-mpnet-base-v2')

img = 'test.jpg'  
result = inference_segmentor(model, img)


arr = np.unique(result[0])
score_1 = np.copy(arr).astype(np.float)
score_2 = np.copy(arr).astype(np.float)

emb_1 = model_bert.encode(caption_1)
emb_2 = model_bert.encode(caption_2)

cos_1 = cos_2 = 0

for i in range(len(arr)):
	# print(label['Name'][arr[i]].split(';')[0])
	emb_label = model_bert.encode(label['Name'][arr[i]].split(';')[0])

	score_1[i] = spatial.distance.cosine(emb_1, emb_label)
	score_2[i] = spatial.distance.cosine(emb_2, emb_label)

print(score_1)
print(score_2)
