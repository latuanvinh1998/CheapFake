import sys
sys.path.append('vqa-maskrcnn-benchmark')

import yaml
import cv2
import torch
import requests
import time
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd

import json
from sentence_transformers import SentenceTransformer, util
from scipy import spatial
import math

import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

import captioning
import captioning.utils.misc
import captioning.models


def E1(TP, TN, FP, FN):

  accuracy = (TP + TN)/(TP + FP + FN + TN)
  precision = TP/(TP + FP)
  recall = TP/(TP + FN)
  f1_score = 2*(recall*precision)/(recall + precision)
  mcc = (TP * TN - FP * FN)/math.sqrt ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

  return accuracy, f1_score, mcc


class FeatureExtractor:
  TARGET_IMAGE_SIZE = [448, 448]
  CHANNEL_MEAN = [0.485, 0.456, 0.406]
  CHANNEL_STD = [0.229, 0.224, 0.225]
  
  def __init__(self):
    # self._init_processors()
    self.detection_model = self._build_detection_model()
  
  def __call__(self, url):
    with torch.no_grad():
      detectron_features = self.get_detectron_features(url)
    
    return detectron_features
  
  def _build_detection_model(self):

      cfg.merge_from_file('model_data/detectron_model.yaml')
      cfg.freeze()

      model = build_detection_model(cfg)
      checkpoint = torch.load('model_data/detectron_model.pth', 
                              map_location=torch.device("cpu"))

      load_state_dict(model, checkpoint.pop("model"))

      model.to("cuda")
      model.eval()
      return model
  
  def get_actual_image(self, image_path):
      if image_path.startswith('http'):
          path = requests.get(image_path, stream=True).raw
      else:
          path = image_path
      
      return path

  def _image_transform(self, image_path):
      path = self.get_actual_image(image_path)

      img = Image.open(path)
      im = np.array(img).astype(np.float32)
      im = im[:, :, ::-1]
      im -= np.array([102.9801, 115.9465, 122.7717])
      im_shape = im.shape
      im_size_min = np.min(im_shape[0:2])
      im_size_max = np.max(im_shape[0:2])
      im_scale = float(800) / float(im_size_min)
      # Prevent the biggest axis from being more than max_size
      if np.round(im_scale * im_size_max) > 1333:
           im_scale = float(1333) / float(im_size_max)
      im = cv2.resize(
           im,
           None,
           None,
           fx=im_scale,
           fy=im_scale,
           interpolation=cv2.INTER_LINEAR
       )
      img = torch.from_numpy(im).permute(2, 0, 1)
      return img, im_scale


  def _process_feature_extraction(self, output,
                                 im_scales,
                                 feat_name='fc6',
                                 conf_thresh=0.2):
      batch_size = len(output[0]["proposals"])
      n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
      score_list = output[0]["scores"].split(n_boxes_per_image)
      score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
      feats = output[0][feat_name].split(n_boxes_per_image)
      cur_device = score_list[0].device

      feat_list = []

      for i in range(batch_size):
          dets = output[0]["proposals"][i].bbox / im_scales[i]
          scores = score_list[i]

          max_conf = torch.zeros((scores.shape[0])).to(cur_device)

          for cls_ind in range(1, scores.shape[1]):
              cls_scores = scores[:, cls_ind]
              keep = nms(dets, cls_scores, 0.5)
              max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                           cls_scores[keep],
                                           max_conf[keep])

          keep_boxes = torch.argsort(max_conf, descending=True)[:100]
          feat_list.append(feats[i][keep_boxes])
      return feat_list
    
  def get_detectron_features(self, image_path):
      im, im_scale = self._image_transform(image_path)
      img_tensor, im_scales = [im], [im_scale]
      current_img_list = to_image_list(img_tensor, size_divisible=32)
      current_img_list = current_img_list.to('cuda')
      with torch.no_grad():
          output = self.detection_model(current_img_list)
      feat_list = self._process_feature_extraction(output, im_scales, 
                                                  'fc6', 0.2)
      return feat_list[0]


def get_captions(img_feature):
    # Return the 5 captions from beam serach with beam size 5
    return model.decode_sequence(model(img_feature.mean(0)[None], img_feature[None], mode='sample', opt={'beam_size':5, 'sample_method':'beam_search', 'sample_n':5})[0])



infos = captioning.utils.misc.pickle_load(open('infos_trans12-best.pkl', 'rb'))

feature_extractor = FeatureExtractor()
infos['opt'].vocab = infos['vocab']

model_bert = SentenceTransformer('paraphrase-mpnet-base-v2').to(torch.device("cuda:0"))
model = captioning.models.setup(infos['opt'])
model.cuda()
model.load_state_dict(torch.load('model-best.pth'))

# captions = get_captions(feature_extractor("input.jpg"))
# for i in range(len(captions)):
#   print(captions[i])

f = open('../Data/mmsys_anns/public_test_mmsys_final.json')
labels = []

# f_cap = open('cap.txt')
# cap = f_cap.read().split('\n')

for line in f:
  labels.append(json.loads(line))

length = len(labels)

label_context = []
thresholds = np.arange(0, 1, 0.01)

idx = 0;

start = time.time()
for label in labels:

  emb_sent_1 = model_bert.encode(label['caption1_modified'])
  emb_sent_2 = model_bert.encode(label['caption2_modified'])
  cos_sim = spatial.distance.cosine(emb_sent_1, emb_sent_2)

  path = '../Data/' +  label['img_local_path']
  path = path.replace(':', '_')

  captions = get_captions(feature_extractor(path))

  emb_sent = model_bert.encode(captions[0])
  print(path)
  # f.write(captions[0] + '\n')
  # emb_sent = model_bert.encode(cap[0])

  cos_1 = spatial.distance.cosine(emb_sent, emb_sent_1)
  cos_2 = spatial.distance.cosine(emb_sent, emb_sent_2)
  label_context.append([label['context_label'], cos_1, cos_2])

  idx += 1

print("Latency: ", time.time() - start)
accs = []
f1s = []
mccs = []

for threshold in thresholds:

  TP = TN = FP = FN = 0
  for label, cos_1, cos_2 in label_context:
    if cos_1 < threshold and cos_2 < threshold and label == 0:
      TN += 1
    elif cos_1 > threshold and cos_2 > threshold and label == 0:
      FP += 1
    elif cos_1 > threshold and cos_2 > threshold and label == 1:
      TP += 1
    elif cos_1 < threshold and cos_2 < threshold and label == 1:
      FN += 1

    elif cos_1 < threshold and cos_2 > threshold and label == 1:
      TP += 1
    elif cos_1 > threshold and cos_2 < threshold and label == 1:
      TP += 1
    elif cos_1 < threshold and cos_2 > threshold and label == 0:
      FP += 1
    elif cos_1 > threshold and cos_2 < threshold and label == 0:
      FP += 1

  if TP != 0 and TN != 0 and FP != 0 and FN != 0:
    acc, f1, mcc = E1(TP, TN, FP, FN)
    accs.append(acc)
    f1s.append(f1)
    mccs.append(mcc)
print("Accuracy: ",accs[np.argmax(np.asarray(accs))])
print("F1 Score: ", f1s[np.argmax(np.asarray(accs))])
print("MCC: ", mccs[np.argmax(np.asarray(accs))])
