# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:44:26 2020

@author: ALW
"""

import tensorflow as tf


def BN(x):
    return tf.layers.batch_normalization(x,axis=-1,momentum=0.99,epsilon=0.001, center=True, scale=True,)
''' 
def Resblock(x,kernel_size,stride,channel):
    layer1=tf.nn.relu(BN(tf.layers.conv2d(x,channel,kernel_size,strides=stride, padding='same')))
    layer1=tf.nn.relu(BN(tf.layers.conv2d(layer1,channel,kernel_size,strides=1, padding='same')))
    skip=tf.layers.conv2d(x,channel,1,strides=stride, padding='same')
    res_value=tf.nn.relu(BN(skip+layer1))
    return res_value

def Dual_attention(x,H,W,C):
    ######position
    x=tf.layers.conv2d(x,C*3,1,strides=1, padding='same')
    Q,K,V_poition=tf.split(x, 3, axis=3)
    Q=tf.reshape(Q,[-1,H*W,C])
    K=tf.reshape(K,[-1,H*W,C])
    V_poition=tf.reshape(V_poition,[-1,H*W,C])
    K=tf.transpose(K,[0,2,1])
    result=tf.matmul(Q,K)
    result=tf.nn.softmax(result)
    V_poition=tf.matmul(result,V_poition)
    V_poition=tf.reshape(V_poition,[-1,H,W,C])
    ######channel
    x=tf.layers.conv2d(x,C*3,1,strides=1, padding='same')
    Q,K,V_channel=tf.split(x, 3, axis=3)
    Q=tf.reshape(Q,[-1,H*W,C])
    K=tf.reshape(K,[-1,H*W,C])
    V_channel=tf.reshape(V_channel,[-1,H*W,C])
    K=tf.transpose(K,[0,2,1])
    result=tf.matmul(K,Q)
    result=tf.nn.softmax(result)
    V_channel=tf.matmul(V_channel,result)
    V_channel=tf.reshape(V_channel,[-1,H,W,C])
    V=V_channel+V_poition
    return V
 

def SK_block(x,kernel1,kernel2,channel):
    ############Spilt
    U1=tf.layers.conv2d(x,channel,kernel1,strides=1, padding='same')
    U2=tf.layers.conv2d(x,channel,kernel2,strides=1, padding='same')
    ############Fuse    
    U=U1+U2
    S=tf.keras.layers.GlobalAvgPool2D()(U)
    print(S)
    S=tf.reshape(S,[-1,1,1,channel])
    print(S)
    Z=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(S,32,1,strides=1, padding='same'),axis=-1,momentum=0.99,epsilon=0.001, center=True, scale=True,))
    print(Z)
    a=tf.layers.conv2d(Z,channel,1,strides=1, padding='same')
    b=tf.layers.conv2d(Z,channel,1,strides=1, padding='same')
    print(a,b)
    combine=tf.concat([a,b],1)
    print(combine)
    combine=tf.nn.softmax(combine,axis=1)
    print(combine)
    a,b=tf.split(combine,num_or_size_splits=2, axis=1)
    print(a,b)
    V=a*U1+b*U2
    print(V)
    return V   



def ASPPV2(x, rate1, rate2, rate3, rate4, channel):
    
    layer1_1=tf.layers.conv2d(x,channel,1,strides=1, padding='same')
    layer1_2=tf.layers.conv2d(x,channel,3,strides=1, padding='same',dilation_rate=rate2)
    layer1_3=tf.layers.conv2d(x,channel,3,strides=1, padding='same',dilation_rate=rate3)
    layer1_4=tf.layers.conv2d(x,channel,3,strides=1, padding='same',dilation_rate=rate4)
    layer1_5=tf.layers.max_pooling2d(x, 3, 1, padding='same')
         
    encoder_output=tf.concat([layer1_1,layer1_2,layer1_3,layer1_4,layer1_5],-1)
    encoder_output=tf.layers.conv2d(encoder_output,channel,1,strides=1, padding='same')
    
    return encoder_output      

'''

def Unet1(x,channel):
    layer1=tf.nn.relu(BN(tf.layers.conv2d(x,64,3,strides=1, padding='same')))
    layer1=tf.nn.relu(BN(tf.layers.conv2d(layer1,64,3,strides=1, padding='same')))
    max_pooling1=tf.layers.max_pooling2d(layer1, 3, strides=2, padding='same') 
    layer2=tf.nn.relu(BN(tf.layers.conv2d(max_pooling1,128,3,strides=1, padding='same')))
    layer2=tf.nn.relu(BN(tf.layers.conv2d(layer2,128,3,strides=1, padding='same')))
    max_pooling2=tf.layers.max_pooling2d(layer2, 3, strides=2, padding='same')
    layer3=tf.nn.relu(BN(tf.layers.conv2d(max_pooling2,128,3,strides=1, padding='same')))
    layer3=tf.nn.relu(BN(tf.layers.conv2d(layer3,128,3,strides=1, padding='same')))    
    print(layer3)
    max_pooling3=tf.layers.max_pooling2d(layer3, 3, strides=2, padding='same')
    layer4=tf.nn.relu(BN(tf.layers.conv2d(max_pooling3,128,3,strides=1, padding='same')))
    layer4=tf.nn.relu(BN(tf.layers.conv2d(layer4,128,3,strides=1, padding='same'))) 
    print(layer4)
    max_pooling4=tf.layers.max_pooling2d(layer4, 3, strides=2, padding='same')
    #layer5=tf.nn.relu(BN(tf.layers.conv2d_transpose(max_pooling4,128,3,strides=2, padding='same')))
    layer5=tf.nn.relu(BN(tf.layers.conv2d(max_pooling4,128,3,strides=1, padding='same')))
    layer5=tf.nn.relu(BN(tf.layers.conv2d(layer5,128,3,strides=1, padding='same'))) 
    print(layer5)
    layer6=tf.nn.relu(BN(tf.layers.conv2d_transpose(layer5,128,3,strides=2, padding='same')))
    print(layer6)
    layer6=tf.concat([layer6,layer4],-1)
    layer6=tf.nn.relu(BN(tf.layers.conv2d(layer6,128,3,strides=1, padding='same')))
    layer6=tf.nn.relu(BN(tf.layers.conv2d(layer6,128,3,strides=1, padding='same')))
    layer7=tf.nn.relu(BN(tf.layers.conv2d_transpose(layer6,128,3,strides=2, padding='same')))   
    print(layer7)
    layer7=tf.concat([layer7,layer3],-1)
    layer7=tf.nn.relu(BN(tf.layers.conv2d(layer7,128,3,strides=1, padding='same')))
    layer7=tf.nn.relu(BN(tf.layers.conv2d(layer7,128,3,strides=1, padding='same')))
    layer8=tf.nn.relu(BN(tf.layers.conv2d_transpose(layer7,128,3,strides=2, padding='same'))) 
    print(layer8)
    layer8=tf.concat([layer8,layer2],-1)
    layer8=tf.nn.relu(BN(tf.layers.conv2d(layer8,128,3,strides=1, padding='same')))
    layer8=tf.nn.relu(BN(tf.layers.conv2d(layer8,128,3,strides=1, padding='same')))
    layer9=tf.nn.relu(BN(tf.layers.conv2d_transpose(layer8,64,3,strides=2, padding='same'))) 
    print(layer9)    
    layer9=tf.concat([layer9,layer1],-1)
    layer9=tf.nn.relu(BN(tf.layers.conv2d(layer9,128,3,strides=1, padding='same')))
    layer9=tf.nn.relu(BN(tf.layers.conv2d(layer9,64,3,strides=1, padding='same')))
    layer9=tf.nn.relu(BN(tf.layers.conv2d(layer9,64,3,strides=1, padding='same'))) 
    layer9=tf.nn.relu(BN(tf.layers.conv2d(layer9,channel,3,strides=1, padding='same')))
    print(layer9)       
    return layer9
    
    
def Unet(x,channel):
    layer1=tf.nn.relu(tf.layers.conv2d(x,64,3,strides=1, padding='same'))
    layer1=tf.nn.relu(tf.layers.conv2d(layer1,64,3,strides=1, padding='same'))
    max_pooling1=tf.layers.max_pooling2d(layer1, 3, strides=2, padding='same') 
    layer2=tf.nn.relu(tf.layers.conv2d(max_pooling1,128,3,strides=1, padding='same'))
    layer2=tf.nn.relu(tf.layers.conv2d(layer2,128,3,strides=1, padding='same'))
    max_pooling2=tf.layers.max_pooling2d(layer2, 3, strides=2, padding='same')
    layer3=tf.nn.relu(tf.layers.conv2d(max_pooling2,128,3,strides=1, padding='same'))
    layer3=tf.nn.relu(tf.layers.conv2d(layer3,128,3,strides=1, padding='same'))    
    print(layer3)
    max_pooling3=tf.layers.max_pooling2d(layer3, 3, strides=2, padding='same')
    layer4=tf.nn.relu(tf.layers.conv2d(max_pooling3,128,3,strides=1, padding='same'))
    layer4=tf.nn.relu(tf.layers.conv2d(layer4,128,3,strides=1, padding='same')) 
    print(layer4)
    max_pooling4=tf.layers.max_pooling2d(layer4, 3, strides=2, padding='same')
    #layer5=tf.nn.relu(BN(tf.layers.conv2d_transpose(max_pooling4,128,3,strides=2, padding='same')))
    layer5=tf.nn.relu(tf.layers.conv2d(max_pooling4,128,3,strides=1, padding='same'))
    layer5=tf.nn.relu(tf.layers.conv2d(layer5,128,3,strides=1, padding='same')) 
    print(layer5)
    layer6=tf.nn.relu(tf.layers.conv2d_transpose(layer5,128,3,strides=2, padding='same'))
    print(layer6)
    layer6=tf.concat([layer6,layer4],-1)
    layer6=tf.nn.relu(tf.layers.conv2d(layer6,128,3,strides=1, padding='same'))
    layer6=tf.nn.relu(tf.layers.conv2d(layer6,128,3,strides=1, padding='same'))
    layer7=tf.nn.relu(tf.layers.conv2d_transpose(layer6,128,3,strides=2, padding='same'))   
    print(layer7)
    layer7=tf.concat([layer7,layer3],-1)
    layer7=tf.nn.relu(tf.layers.conv2d(layer7,128,3,strides=1, padding='same'))
    layer7=tf.nn.relu(tf.layers.conv2d(layer7,128,3,strides=1, padding='same'))
    layer8=tf.nn.relu(tf.layers.conv2d_transpose(layer7,128,3,strides=2, padding='same')) 
    print(layer8)
    layer8=tf.concat([layer8,layer2],-1)
    layer8=tf.nn.relu(tf.layers.conv2d(layer8,128,3,strides=1, padding='same'))
    layer8=tf.nn.relu(tf.layers.conv2d(layer8,128,3,strides=1, padding='same'))
    layer9=tf.nn.relu(tf.layers.conv2d_transpose(layer8,64,3,strides=2, padding='same')) 
    print(layer9)    
    layer9=tf.concat([layer9,layer1],-1)
    layer9=tf.nn.relu(tf.layers.conv2d(layer9,128,3,strides=1, padding='same'))
    layer9=tf.nn.relu(tf.layers.conv2d(layer9,64,3,strides=1, padding='same'))
    layer9=tf.nn.relu(tf.layers.conv2d(layer9,64,3,strides=1, padding='same'))
    layer9=tf.nn.relu(tf.layers.conv2d(layer9,channel,3,strides=1, padding='same'))
    print(layer9)       
    return layer9
    
    
    
    