# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:03:24 2018

@author: zhupc
"""
import tensorflow as tf
import numpy as np
import model.vgg16_structure as vgg
from functools import reduce
from model.activation import Activation
class VGG16:
    
    def __init__(self,weights_path):
        
        self.WEIGHTS_PATH=weights_path
        
        
    def load_initial_weights(self, session):
        '''
        初始化权重
        '''
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()   
        print('weights_dict',weights_dict.keys())           
        for op_name in weights_dict:
            with tf.variable_scope(op_name,reuse=True):
                for data in weights_dict[op_name]:
                    if len(data.shape)==1:
                        var=tf.get_variable('biases',trainable=False)
                        session.run(tf.assign(var,data))
                    else:
                        var=tf.get_variable('weights',trainable=False)
                        session.run(var.assign(data))
    
    def build(self, X,is_training=True):
            self.train_phase=tf.constant(is_training) if is_training else None
            
            self.conv1_1=self.convolution(X,'conv1_1')
            self.conv1_2=self.convolution(self.conv1_1,'conv1_2')
            self.pool1=self.pooling(self.conv1_2,'pool1')
            
            self.conv2_1=self.convolution(self.pool1,'conv2_1')
            self.conv2_2=self.convolution(self.conv2_1,'conv2_2')
            self.pool2=self.pooling(self.conv2_2,'pool2')
            
            self.conv3_1=self.convolution(self.pool2,'conv3_1')
            self.conv3_2=self.convolution(self.conv3_1,'conv3_2')
            self.conv3_3=self.convolution(self.conv3_2,'conv3_3')
            self.pool3=self.pooling(self.conv3_3,'pool3')
            
            self.conv4_1=self.convolution(self.pool3,'conv4_1')
            self.conv4_2=self.convolution(self.conv4_1,'conv4_2')
            self.conv4_3=self.convolution(self.conv4_2,'conv4_3')
            self.pool4=self.pooling(self.conv4_3,'pool4')
            
            self.conv5_1=self.convolution(self.pool4,'conv5_1')
            self.conv5_2=self.convolution(self.conv5_1,'conv5_2')
            self.conv5_3=self.convolution(self.conv5_2,'conv5_3')
            self.pool5=self.pooling(self.conv5_3,'pool5')
            
            self.fc6=self.fully_connection(self.pool5,Activation.relu,'fc6')
            

            self.fc7 = self.fully_connection(self.fc6, Activation.relu, 'fc7')
            self.fc8 = self.fully_connection(self.fc7, Activation.relu, 'fc8')
           
            self.prob = self.fc8

            print('prob shape',self.prob.get_shape())
            return self.prob
        
        
    def fully_connection(self,X,activation,name):
        size=vgg.structure[name]
        with tf.variable_scope(name) as scope:
             shape=X.get_shape().as_list()
             print(name,shape)
             dim=reduce(lambda x,y:x*y,shape[1:])
             
             x = tf.reshape(X, [-1, dim])
             
             weights=self.get_weight([dim,size[0][0]],name='weights')
             biase=self.get_biase(size[1],name='biases')
             fc=tf.nn.bias_add(tf.matmul(x,weights),bias=biase,name=scope.name)
             fc=activation(fc)
             print('input shape is :'+str(shape))
             print('total nuron count is :'+str(dim))
             return self.batch_normalization(fc)
    
    def pooling(self,X,name):
        return tf.nn.max_pool(X,ksize=vgg.ksize,strides=vgg.pool_strides,padding='SAME',name=name)
    
    
    def convolution(self,X,name):
        '''
        卷积操作
        '''
        print('conv size is :'+str(X.get_shape().as_list()))
        with tf.variable_scope(name) as scope:
            size=vgg.structure[scope.name]
            kernel=self.get_weight(size[0],name='weights')
            biase=self.get_biase(shape=size[1],name='biases')
            conv=tf.nn.conv2d(X,filter=kernel,strides=vgg.conv_strides,
                              padding='SAME',name=scope.name)
            out= tf.nn.relu(tf.nn.bias_add(conv,bias=biase))
        return self.batch_normalization(out) 
        
    def get_weight(self,shape,name):
        
        return tf.get_variable(shape=shape,
                            
                               name=name)
    def get_biase(self,shape,name):
        
        return tf.get_variable(shape=shape,name=name)
    
    
    def batch_normalization(self,input, decay=0.9, eps=1e-5):
        """
        Batch Normalization
        Result in:
            * Reduce DropOut
            * Sparse Dependencies on Initial-value(e.g. weight, bias)
            * Accelerate Convergence
            * Enable to increase training rate

        Args: output of convolution or fully-connection layer
        Returns: Normalized batch
        """
        shape = input.get_shape().as_list()
        n_out = shape[-1]
        beta = tf.Variable(tf.zeros([n_out]))
        gamma = tf.Variable(tf.ones([n_out]))

        if len(shape) == 2:
            batch_mean, batch_var = tf.nn.moments(input, [0])
        else:
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2])

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(tf.Variable(True), mean_var_with_update,
          lambda: (ema.average(batch_mean), ema.average(batch_var)))

        return tf.nn.batch_normalization(input, mean, var, beta, gamma, eps)

        
        