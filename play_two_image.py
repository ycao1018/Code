

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import scipy.io as sio
import glob
import os
from time import sleep

def show_two_image(image1,image2,num,pause_time):


    count=0
    fig=plt.figure()
    for time in range(0,num):
        ax1=fig.add_subplot(121)
        #plt.imshow(image1[:,:,time],image2[:,:,time], cmap= None)
        ax1.imshow(image1[:,:,time],cmap='Greys')
        
        ax2=fig.add_subplot(122)
        ax2.imshow(image2[:,:,time],cmap='Greys')
        #plt.imshow(image[:,:,time])
        plt.pause(pause_time)
        count+=1
    plt.show(block=True)

    
