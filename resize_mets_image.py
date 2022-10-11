

import numpy as np
import PIL
from PIL import Image



def resize_img(image):
    vol=80
    [row,col,de]=image.shape
    img=np.zeros([vol,vol,de])
    for k in range(0,de):
        im=image[:,:,k]
        im.shape
        im_pil=np.array(Image.fromarray(im).resize((80, 80), Image.NEAREST))
        img[:,:,k]=im_pil
        imag=img
    return imag
        
