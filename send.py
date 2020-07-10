import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as pyplot
import os
 
dir="/home/liuxina/Downloads/cycada_release/data/images/"#npy文件路径
dest_dir="/home/liuxina/Downloads/cycada_release/data/ct_img/"
def npy2jpg(dir,dest_dir):
    if os.path.exists(dir)==False:
        os.makedirs(dir)
    if os.path.exists(dest_dir)==False:
        os.makedirs(dest_dir)

'''
    file=dir+'gt.npy'
    con_arr=np.load(file)
'''

i=0
for i in range(8400):
        mrr=str(i)+'_real_B.png'
        file=dir+mrr
        arr=np.array(Image.open(file))
        misc.imsave(dest_dir+"_"+str(i)+".png" , arr)
        i=i+1

'''
    count=0
    for con in con_arr:
        arr=con[0]
        label=con[1]
        print(np.argmax(label))
        arr=arr*255
        #arr=np.transpose(arr,(2,1,0))
        arr=np.reshape(arr,(256,256))
        r=Image.fromarray(arr[0]).convert("L")
        g=Image.fromarray(arr[1]).convert("L")
        b=Image.fromarray(arr[2]).convert("L")
 
        img=Image.merge("RGB",(r,g,b))
 
        label_index=np.argmax(label)
        img.save(dest_dir+str(label_index)+"_"+str(count)+".png")
        count=count+1
'''
 
npy2jpg(dir,dest_dir)
