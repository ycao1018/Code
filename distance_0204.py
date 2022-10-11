

import os
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
#from unet_ming.resize_mets_image import resize_img



def load_images(folderpath):
    folderlist=os.listdir(folderpath)
    folderlist=np.sort(folderlist)
    num=np.size(folderlist)
    imag=np.zeros([num,200,200,175],dtype='float16')
    for k in range(0,num):

        filepath=os.path.join(folderpath,folderlist[k])

        img=nib.load(filepath)
        datat1=img.get_fdata()
        
        datat2=datat1[80:280,170:370,100:275]
        imag[k,:,:,:]=datat2
            
        #imag.append(datat1)
    return imag


path="/home/yufeng/Desktop/data/data need later"
#train_x,yy=data_meni_training(path)
imag=load_images(path)
imag1=load_images(path)
print(imag.shape)


from scipy.spatial.distance import cdist, squareform
from scipy import ndimage

imag[imag!=1]=0

imag1[imag1==1]=0
imag1[imag1!=0]=1
import tensorflow as tf
imag=tf.constant(imag)
imag1=tf.constant(imag1)

def gpu_cal(imag,imag1):
    
    for k in range(6,81):
        print(k)
        m1=imag[k,:,:,:]
        #m1[m1!=1]=0
        aa,ab,ac=np.where(m1==1)
        m2=imag1[k,:,:,:]
        #m2[m2==1]=0
        #m2[m2!=0]=1
        da,db,dc=np.where(m2==1)
        dist=[]
        for i in range(len(aa)):
            
            for j in range(len(da)):
                
            #p1=np.array(aa[i],ab[i],ac[i])
            #p2=np.array(da[j],db[j],dc[j])
            #vv=np.sum((p1-p2)**2, axis=0)
            #vb=np.sqrt(vv)
                vv=np.sqrt((aa[i]-da[j])**2+(ab[i]-db[j])**2+(ac[i]-dc[j])**2)
                dist.append(vv)
        mb=np.min(dist)
        print(mb)
    return mb


strategy=tf.distribute.MirroredStrategy(["/GPU:0", "/GPU:1", "/GPU:2"])
#strategy=tf.distribute.MirroredStrategy(["/GPU:2", "/GPU:3"])
with strategy.scope():
    
    cal=gpu_cal(imag,imag1)














    


