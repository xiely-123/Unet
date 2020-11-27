# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:29:58 2020

@author: ALW
"""

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import dataread



def IOU(label,Prediction):
    U = 0
    I = 0
    x,y = label.shape
    for i in range(x):
        for j in range(y):
            if label[i,j] == Prediction[i,j] & label[i,j]!=0:
                U=U+1
                I=I+1
            if label[i,j] != Prediction[i,j]:
                U=U+1
    IOU_value = I/U
    return IOU_value

def test_IOU(label,Prediction,num):
    test_IOU_value=0
    for i in range(num):
        label_1=label[i,:,:]
        label_1=label_1.reshape((512,512))
        prediction_1=Prediction[i,:,:]
        prediction_1=prediction_1.reshape((512,512))
        value = IOU(label_1, prediction_1)    
        test_IOU_value = test_IOU_value + value
    return test_IOU_value/num     


image_dir="D:/机器学习与深度学习/Segment_net/dataset/test_image/"
label_dir="D:/机器学习与深度学习/Segment_net/dataset/test_label/"
#image_dir="D:/机器学习与深度学习/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/validation/"
#label_dir="D:/机器学习与深度学习/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/annotations/validation/"
# image = Image.open("D:/工作/test_data_jpg/30.jpg")
# im_array = np.array(image)
# image_batch = im_array/np.max(im_array)
# image_batch=image_batch.reshape((-1,512,512,1))


sess=tf.Session()
saver = tf.train.import_meta_graph( './Mobel/best.ckpt.meta')# 加载图结构
saver.restore(sess, "./Mobel/best.ckpt")
gragh = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
#tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]# 得到当前图中所有变量的名称
#print(tensor_name_list)
x = gragh.get_tensor_by_name('Placeholder:0')
#y = gragh.get_tensor_by_name('Placeholder_1:0')
feature = gragh.get_tensor_by_name('Relu_23:0')
softmax_feature=tf.argmax(feature,axis=-1)
print("aaaaaaaaaaaaa")
print(softmax_feature)
print(feature)
IOU_all=0
###测试
for i in range(100):
    image_batch, label_batch = dataread.readdata(image_dir,label_dir,15)
    #image_batch, label_batch = dataread.readdata(image_dir,label_dir,1)
    image_batch = image_batch.reshape((-1,512,512,1))
    #softmax_feature_1=sess.run([softmax_feature], feed_dict={x: image_batch, y: label_batch})
    softmax_feature_1=sess.run([softmax_feature], feed_dict={x: image_batch})
    #test=sess.run([feature], feed_dict={x: image_batch, y: label_batch})
    #print()
    
    #print(softmax_feature)
    #test_result = sess.run([test_result], feed_dict={x: image_batch, y: label_batch})
    
    picture=np.array(softmax_feature_1).reshape((-1,512,512))
    print(picture.shape)
    label_batch=np.array(label_batch).reshape((-1,512,512))
    test=test_IOU(label_batch, picture,15)
    print("test_IOU : "+str(test)) 
    IOU_all=IOU_all+test
IOU_all=IOU_all/100
print("IOU_all : "+str(IOU_all))    
##测试结束
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
'''
###显示
image_batch = np.reshape(image_batch, [512,512,1])
plt.figure()
plt.subplot(1,3,1)
plt.title('Origin') 
plt.imshow(image_batch, interpolation = 'bilinear')
plt.subplot(1,3,2)
plt.title('Groundtruth') 
plt.imshow(label_batch, interpolation = 'bilinear')
plt.subplot(1,3,3)
plt.title('Prediction') 
plt.imshow(picture, interpolation = 'bilinear')
plt.show()
#####
print(image_batch.shape)
#print(label_batch)
#print(test_result)
'''