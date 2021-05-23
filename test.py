from torch import nn
from sentence_transformers import SentenceTransformer, util
from neurnet import *

import torch

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

anchor = torch.randn(100, 128)
positive = torch.randn(100, 128)
negative = torch.randn(100, 128)

nn_sentence = Caption_Emb(768, 256)

# output = triplet_loss(anchor, positive, positive)
# print(output)

model_nlp = SentenceTransformer('stsb-mpnet-base-v2')
sent_emb_1 = torch.flatten(torch.tensor(model_nlp.encode("Hello Darkness My Friend")))
print(nn_sentence(sent_emb_1).shape)