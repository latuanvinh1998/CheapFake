import numpy as np

import math
import torch
import random

from PIL import Image
from torchvision import transforms
from torch import nn


def E1(TP, TN, FP, FN):

	accuracy = (TP + TN)/(TP + FP + FN + TN)
	precision = TP/(TP + FP)
	recall = TP/(TP + FN)
	f1_score = 2*(recall*precision)/(recall + precision)
	mcc = (TP * TN - FP * FN)/math.sqrt ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

	return accuracy, f1_score, mcc, precision, recall
