

import os
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
from unet_met.resize_mets_image import resize_img



def load_images(folderpath):
    folderlist=os.listdir(folderpath)
    folderlist=np.sort(folderlist)
    num=np.size(folderlist)
    imag=np.zeros([num,80,80,160],dtype='float16')
    for k in range(0,num):

        filepath=os.path.join(folderpath,folderlist[k])
        img=nib.load(filepath)
        datat1=img.get_fdata()
        datat1=datat1[20:340,110:430,115:275] 
        #datat1=datat1[80:280,170:370,115:275] 
        datat1[datat1>2]=2
        #datat1[datat1!=0]=1
        datat2=resize_img(datat1)
        imag[k,:,:,:]=datat2
            
        #imag.append(datat1)
    #img=imag[:,0:80,0:80,20:140]
    return imag





