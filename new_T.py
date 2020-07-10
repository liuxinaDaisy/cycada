import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as pyplot
import os

dir = "/home/liuxina/Downloads/cycada_release/data_npy/"
a=np.ones((1024,3,256,256))

mrr='ct_img.npy'
file=dir+mrr
arr=np.load(file)


for i in range(1024):
        a[i] = arr[i].T
        a[i][0]=a[i][0].T
        a[i][1]=a[i][1].T
        a[i][2]=a[i][2].T
        #a[i]=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        i=i+1


np.save('/home/liuxina/Downloads/cycada_release/data_npy/new_img/ct_img.npy',a)


