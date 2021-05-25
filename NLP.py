from sentence_transformers import SentenceTransformer, util
from scipy import spatial
import numpy as np

model = SentenceTransformer('stsb-mpnet-base-v2')

sentence_1 = "building"
sentence_2 = "Supporters of Tanzania's ruling Chama Cha Mapinduzi party come out on Friday to celebrate their candidate's victory in the disputed Zanzibari presidential election"

emb_1 = model.encode(sentence_1)

emb_2 = model.encode(sentence_2)

# cos_sim = util.pytorch_cos_sim(emb_1, emb_2)
result = spatial.distance.cosine(emb_1, emb_2)
print("Cosine-Similarity:", result)