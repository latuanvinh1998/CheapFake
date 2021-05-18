from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('stsb-mpnet-base-v2')

sentence_1 = "Hello Darkness My Friend"
sentence_2 = "Computer Vision"

emb_1 = model.encode(sentence_1)

emb_2 = model.encode(sentence_2)

cos_sim = util.pytorch_cos_sim(emb_1, emb_2)
print("Cosine-Similarity:", cos_sim[0][0].numpy())