import os
import numpy as np

import time

start=time.time()

import matplotlib.pyplot as plt
from tensorflow import keras
from unet_met.cao_early_bird import cao2019_model

n_epochs=300
steps_per_epoch=3
validation_steps=1
batch_size=6

path="/home/yufeng/Desktop/data/ct data ptv stem"
##path3='C:/Users/ycao1/Documents\meningioma/data728'
#from unet_lgg.data_generator_lgg import data_lgg_training
from unet_met.data_generator_met import load_images
##from unet_ming.data_generator_meni_three import data_meni_training_three

bat=66
vol=80
train_data=load_images(path)
train_da=np.reshape(train_data,[bat,1,vol,vol,160])
train_d=np.float32(train_da)
##train_data_three=data_meni_training_three(path3,17)
print(train_d.shape)
x_train=train_d[0:bat,:,:,:]
##x_test=train_data_three[0:10,0:1,:,:,:]

y_train=np.zeros([bat,1])
y_train[[3,4,5,8,9,10,11,12,14,20,23,25,27,28,33,36,38,39,41,42,44,45,47,53,62,65],:]=1
#y_train[[3,4,5,8,9,10,11,12,14,20,23,25,27,28],:]=1
y_train=keras.utils.to_categorical(y_train,2)

y_test=np.zeros([10,1])
y_test[[0,5,6,7,8],:]=1
y_test=keras.utils.to_categorical(y_test,2)

#validation_data=(x_train,y_train)

model = cao2019_model(input_shape=[1,vol,vol,160], n_labels=2, initial_learning_rate=5e-6, n_base_filters=10)

print(5+5)
mo=model.fit(x_train,y_train,batch_size=batch_size,epochs=n_epochs)

#model.fit(x_train,y_train,batch_size=batch_size,epochs=n_epochs,validation_data=(x_test,y_test))

#ww=model.predict(mm)
#from Unet_souce.play_image import show_image

path2="/home/yufeng/Desktop/data/ct data ptv stem mo"
test_da=load_images(path2)
test_data=np.reshape(test_da,[20,1,vol,vol,160])
for k in range(0,20):
    mm1=test_data[k,:,:,:,:]
    mm1=np.reshape(mm1,[1,1,vol,vol,160])
    w11=model.predict(mm1)
    print(w11)
