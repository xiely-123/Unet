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



#image_dir="D:/机器学习与深度学习/Segment_net/dataset/image/"
#label_dir="D:/机器学习与深度学习/Segment_net/dataset/label/"
image_dir="D:/机器学习与深度学习/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/validation/"
label_dir="D:/机器学习与深度学习/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/annotations/validation/"

sess=tf.Session()
saver = tf.train.import_meta_graph( './Mobel/best.ckpt.meta')# 加载图结构
saver.restore(sess, "./Mobel/best.ckpt")
gragh = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
#tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]# 得到当前图中所有变量的名称
#print(tensor_name_list)
x = gragh.get_tensor_by_name('Placeholder:0')
y = gragh.get_tensor_by_name('Placeholder_1:0')
feature = gragh.get_tensor_by_name('Relu_23:0')
softmax_feature=tf.argmax(feature,axis=-1)
print(softmax_feature)


print(feature)

image_batch, label_batch = dataread.readdata(image_dir,label_dir,1)
#image_batch = image_batch.reshape((-1,224,224,3))
softmax_feature_1=sess.run([softmax_feature], feed_dict={x: image_batch, y: label_batch})
test=sess.run([feature], feed_dict={x: image_batch, y: label_batch})
#print()
print("aaaaaaaaaaaaa")
#print(softmax_feature)
#test_result = sess.run([test_result], feed_dict={x: image_batch, y: label_batch})

picture=np.array(softmax_feature_1).reshape((224,224))


print(picture.shape)
label_batch=label_batch.reshape(224,224)
m,n=label_batch.shape
print(m,n)
count=0
for i in range(m):
    for j in range(n):
       if label_batch[i,j]==picture[i,j]:
           count=count+1
print("IOU = ",100*count/(m*n))           
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
###显示
image_batch = np.reshape(image_batch, [224,224,3])
plt.figure()
plt.subplot(1,3,1)
plt.title('Origin') 
plt.imshow(image_batch, interpolation = 'bilinear')
plt.subplot(1,3,2)
plt.title('Groudtruth') 
plt.imshow(label_batch, interpolation = 'bilinear')
plt.subplot(1,3,3)
plt.title('Prediction') 
plt.imshow(picture, interpolation = 'bilinear')
plt.show()
#####
print(image_batch.shape)
print(label_batch)
#print(test_result)
