# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:39:48 2020

@author: ALW
"""

import numpy as np
import scipy.misc as misc
import os, random, shutil
from PIL import Image





#image_dir="D:/机器学习与深度学习/Segment_net/dataset/image/"
#label_dir="D:/机器学习与深度学习/Segment_net/dataset/label/"

'''
def readdata(image_dir,label_dir,batch_size):
        data=[]
        label=[]
        pathDir = os.listdir(image_dir)   
        #print(pathDir)#取图片的原始路径
        filenumber=len(pathDir)
        #print(filenumber)
        sample = random.sample(pathDir, batch_size)  
        #print (sample)

        for name in sample:
               image = Image.open(image_dir+name)
               im_array = np.array(image)
               im_array = im_array/np.max(im_array)
               #print(name,im_array.shape)
               name=name.replace('img','label')
               label_data = Image.open(label_dir+name)
               la_array = np.array(label_data)
               #print(name,la_array.shape)
               #np.savetxt("a.txt",la_array)
               data.append(im_array)
               label.append(la_array)
               
        return np.array(data), np.array(label)
'''
#ADE_val_00000001   ADE_val_00000001
#a,b=readdata(image_dir,label_dir,1)
#print(a.shape,b.shape)
image_dir="D:/机器学习与深度学习/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/validation/"
label_dir="D:/机器学习与深度学习/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/annotations/validation/"
def readdata(image_dir,label_dir,batch_size):
        data=[]
        label=[]
        pathDir = os.listdir(image_dir)   
        #print(pathDir)#取图片的原始路径
        filenumber=len(pathDir)
        #print(filenumber)
        sample = random.sample(pathDir, batch_size)  
        #print (sample)

        for name in sample:
               image = Image.open(image_dir+name)
               im_array = np.array(image)
               im_array = im_array/np.max(im_array)
               #im_array = misc.imresize(im_array, size=(224, 224, 3))
               #print(name,im_array.shape)
               name=name.replace('image','label')
               label_data = Image.open(label_dir+name)
               la_array = np.array(label_data)
               #la_array = misc.imresize(la_array, size=(224, 224))               
               #print(name,la_array.shape)
               #np.savetxt("a.txt",la_array)
               data.append(im_array)
               label.append(la_array)
               
        return np.array(data), np.array(label)
#a,b=readdata(image_dir,label_dir,5)
#print(a.shape,b.shape)