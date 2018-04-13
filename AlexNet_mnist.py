#encoding:utf-8 
"""
DtatSet : Mnist
Network model : AlexNet-6
Order:
    1.Accuracy up to 98%
    2.Output result of True and Test labels
    3.Output image and label
    4.ues tf.nn.**  funcation
Time : 2018/04/12
Author:zswang

"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
#read data set
mnist = input_data.read_data_sets("MNIST_Data", one_hot=True)

#define super parameter
learning_rate = 0.0045
batch_size = 128
epoch = 1200
train_keep_prop = 0.95
#define parameter
display_step = 200
classs_num = 10
image_size = 784

#define placceholder to train in sess
xs = tf.placeholder(tf.float32,[None,image_size])		
ys = tf.placeholder(tf.float32,[None,classs_num])		
keep_prop = tf.placeholder(tf.float32)

#read test dataset
test_x = mnist.test.images[:1000]
test_y = mnist.test.labels[:1000]

#image preprocessing to reshape
x_image = tf.reshape(xs,[-1,28,28,1])	

#define model funcation  
    #define varibale parameter
def weight_variable(shape):
    return tf.Variable(tf.random_normal(shape))
def bias_variable(shape):
    return tf.Variable(tf.random_normal(shape))    
    #conv layer
def con2d(input,weight,bias):              #weight = kenel size
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input,weight,strides=[1,1,1,1],padding='SAME'),bias))
    #pool layer
def max_pool(input,ksize):                 #ksize = kenel                 
    return tf.nn.max_pool(input,ksize=[1,ksize,ksize,1],strides=[1,ksize,ksize,1],padding='SAME')
    #batch-normal
def batch_normal(input,lsize = 4):       #in furture can add more parameter
    return tf.nn.lrn(input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    #drop layer
def drop(input,keep_prop):
    return tf.nn.dropout(input,keep_prop)
#creat alexnet model                                    
    #define parameter for weight and bias 
weight = {
        'wc1':weight_variable([3,3,1,64]),
        'wc2':weight_variable([3,3,64,128]),
        'wc3':weight_variable([3,3,128,256]),
        'wd1':weight_variable([4096,1024]),
        'wd2':weight_variable([1024,256]),
        'out':weight_variable([256,classs_num])
}
bias = {
        'bc1':bias_variable([64]),
        'bc2':bias_variable([128]),
        'bc3':bias_variable([256]),
        'bd1':bias_variable([1024]),
        'bd2':bias_variable([256]),
        'out':bias_variable([classs_num])        
}
    #optimizing policy: change parameter model to dictnory
    #conve layer
conv1 = con2d(x_image, weight['wc1'], bias['bc1'])
pool1 = max_pool(conv1,ksize=2)
norm1 = batch_normal(pool1,lsize = 4)
L1 = drop(norm1,keep_prop = keep_prop)

conv2 = con2d(L1, weight['wc2'], bias['bc2'])
pool2 = max_pool(conv2, ksize = 2)
norm2 = batch_normal(pool2, lsize=4)
L2 = drop(norm2, keep_prop)

conv3 = con2d(L2, weight['wc3'], bias['bc3'])
pool3 = max_pool(conv3, ksize=2)
norm3 = batch_normal(pool3, lsize=4)
L3 = drop(norm3, keep_prop)  

    #flaten layer
flat = tf.reshape(L3,[-1,4096]) 
    #dense layer
dense1 = tf.nn.relu(tf.matmul(flat,weight['wd1'])+ bias['bd1'])  
dense2 = tf.nn.relu(tf.matmul(dense1,weight['wd2']) + bias['bd2'])
prediction =tf.matmul(dense2,weight['out']) + bias['out']

#computer loss,As target funcation
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction) 
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
#define gradent dencent model to minimize loss(target funcation)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#computer accuracy
accury = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1)),tf.float32))
init = tf.global_variables_initializer() 
with tf.Session() as sess:
    sess.run(init) 
    for i in range (epoch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict = {xs : batch_xs,ys:batch_ys,keep_prop:train_keep_prop})
        if i%display_step == 0:
            loss = sess.run(cross_entropy,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prop:train_keep_prop})
            train_accuracy = sess.run(accury,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prop:train_keep_prop})
            test_accuracy = sess.run(accury,feed_dict = {xs:test_x,ys:test_y,keep_prop:1})
            print('Loss : '+ str(loss) + ' | Train Accuracy : '+ str(train_accuracy) + ' | Test Accuracy : ' +  str(test_accuracy))                                                               
    for j in range(1):
        print("--------------------Compare to True and Test----------------------------")
        plt.imshow(test_x[j].reshape((28,28)), cmap='gray') 
        plt.show()
        print("True label ："+str(np.argmax(test_y[0:j+1],1)))
        pre_prop = sess.run(prediction,{xs:test_x[0:j+1],keep_prop:1})
        print("Test label ："+str(np.argmax(pre_prop,1)))