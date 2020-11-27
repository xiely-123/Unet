from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import datetime
from six.moves import xrange
import model 
import os
import dataread

#image_dir="D:/机器学习与深度学习/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/validation/"
#label_dir="D:/机器学习与深度学习/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/annotations/validation/"

image_dir="D:/机器学习与深度学习/Segment_net/dataset/image/"
label_dir="D:/机器学习与深度学习/Segment_net/dataset/label/"



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
max_steps = 100000





def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)





image_holder = tf.placeholder(tf.float32, [None, 512, 512, 1])
label_holder = tf.placeholder(tf.int32, [None, 512, 512])

#W8 = weight_variable([3, 3, 1, 2], name="W8")
#b8 = bias_variable([2], name="b8")
#layer1 = tf.nn.relu(conv2d_basic(image_holder, W8, b8))
layer1=model.Unet(image_holder,2)

print(layer1)


def loss(logits, labels):
    '''计算CNN的loss
    tf.nn.sparse_softmax_cross_entropy_with_logits作用：
    把softmax计算和cross_entropy_loss计算合在一起'''
    labels = tf.cast(labels, tf.int64)
    #logits = tf.argmax(logits,axis=-1)
    print("aaaaaaaaaaaaaaaaaaaaaaa")
    print(logits,labels)
    print("aaaaaaaaaaaaaaaaaaaaaaa")
  
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    print(cross_entropy)
    # tf.reduce_mean对cross entropy计算均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    # tf.add_to_collection:把cross entropy的loss添加到整体losses的collection中
    tf.add_to_collection('losses', cross_entropy_mean)
    # tf.add_n将整体losses的collection中的全部loss求和得到最终的loss
    return tf.add_n(tf.get_collection('losses'), name='total_loss') ,cross_entropy

loss, CE = loss(layer1, label_holder)
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
# 求输出结果中top k的准确率，默认使用top 1(输出分类最高的那一类的准确率)
#top_k_op = tf.nn.in_top_k(layer1, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
all_time=0
saver = tf.train.Saver()
basic_acc=0

for step in range(max_steps):
    start_time = time.time() 
    # 获得一个batch的训练数据
    image_batch, label_batch = dataread.readdata(image_dir,label_dir,3)
    image_batch = image_batch.reshape((-1,512,512,1))
    # 将batch的数据传入train_op和loss的计算
    
#    _, loss_value, CE_value, W8_value = sess.run([train_op, loss, CE, W8],
#                            feed_dict={image_holder: image_batch, label_holder: label_batch})

    _, loss_value, CE_value = sess.run([train_op, loss, CE],
                             feed_dict={image_holder: image_batch, label_holder: label_batch})
    #np.savetxt("loss.txt",np.array(CE_value).reshape((512,512)))
   # print(W8_value)
    train_time = time.time() - start_time
    #print(train_time)
    all_time = all_time + train_time
    #print(step,all_time)
    #duration = time.time() - start_time
    if step % 10 == 0:
        start_time_test = time.time()

        image_batch, label_batch = dataread.readdata(image_dir,label_dir,1)
        # 计算这个batch的top 1上预测正确的样本数
        
        test_time = time.time() - start_time_test
        
        all_time = all_time + test_time
        format_str = ('step %d, loss=%.4f (train_batch_time = %.2f sec, test_time = %.2f sec, all_time = %.2f sec)')
        print(format_str % (step, loss_value, train_time, test_time, all_time)) 
        saver.save(sess, './Mobel/best.ckpt')
        all_time = 0
        


























