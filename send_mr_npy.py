import numpy as np
import imageio
import os
from PIL import Image
from scipy import misc
import matplotlib.pyplot as pyplot
import cv2

os.chdir('/home/liuxina/Downloads/cycada_release/data/mr/mr_img')
a=np.ones((1024,256,256,3))

i=0
dir="/home/liuxina/Downloads/cycada_release/data/mr/mr_img/"
for i in range(1024):
        mrr='_'+str(i)+'.png'
        file=dir+mrr
        arr=np.array(Image.open(file))
        a[i] = arr
        #a[i]=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        i=i+1


np.save('/home/liuxina/Downloads/cycada_release/data_npy/mr_img.npy',a)

