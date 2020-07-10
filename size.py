import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as pyplot
import os

dir = "/home/liuxina/Downloads/cycada_release/data_npy/"

mrr='ct_gt.npy'
file=dir+mrr
arr=np.load(file)
print(arr.shape)
