# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:24:02 2018

@author: zhupc
"""

import os

import tensorflow as tf  
import vgg16
from datetime import datetime

def load_dataset(filepath):
    
    def decode(serialized_example):
        features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
        )
        label = features['label']
        img = features['img_raw']
        img = tf.decode_raw(img, tf.uint8)
        img = tf.reshape(img, [227, 227, 3])
        #归一化
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        label = tf.cast(label-1, tf.int32)
        return img, label
        
    dataset=tf.data.TFRecordDataset(filenames=filepath)
    dataset=dataset.map(decode).shuffle(2000)
    return dataset
    



num_epochs = 200
batch_size = 20

train_batches_per_epoch=3215
valid_epoch=357
#每10步，把tf.summary data写入磁盘
display_step = 10

filewriter_path = "tensorboard"
checkpoint_path = "checkpoints"

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
    
    
    
#定义palceholder
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3],name='fetures')
y = tf.placeholder(tf.int64, [batch_size],name='labels')
vgg_model=vgg16_copy.VGG16('weight/vgg_params.npy')
#预测值结果
prob=vgg_model.build(x,is_training=True)

#损失函数
with tf.name_scope('cross_entrpy'):
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob,labels=y))


#学习率指数衰减
valid_step=tf.Variable(1,name='valid_step')

global_step=tf.Variable(0,name='global_step')
learn_rate=tf.train.exponential_decay(0.001,
                                      global_step=global_step,
                                      decay_steps=train_batches_per_epoch,decay_rate=0.95,
                                      staircase=True,name='learn_rate')



with tf.name_scope('train'):
    train_op=tf.train.AdamOptimizer(learn_rate).minimize(loss)


#可视化损失值
tf.summary.scalar('cross_entropy',loss)

#模型准确度
with tf.name_scope('accuracy'):
    correct_pred=tf.nn.in_top_k(prob,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32),name='accuracy')
    
tf.summary.scalar('accuracy',accuracy)

#合并summary,合并之后，执行合并后的结果，就都会被执行
summary_all=tf.summary.merge_all()

#初始化，FileWriter
writer=tf.summary.FileWriter(filewriter_path)

#初始化保存模型对象
saver=tf.train.Saver()

train_dataset=load_dataset("train/train.tfrecords").batch(batch_size)
valid_dataset=load_dataset("valid/valid.tfrecords").batch(batch_size)

iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
train_iterator=iterator.make_initializer(train_dataset,name='trian_iterator')
valid_iterator=iterator.make_initializer(valid_dataset,name='trian_iterator')
next_element=iterator.get_next()

#设置GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
##设置按需增长
#config=tf.ConfigProto(allow_soft_placement=True)
config = tf.ConfigProto(allow_soft_placement=True) 
config.gpu_options.allow_growth = True 

#config.gpu_options.per_process_gpu_memory_fraction = 0.7 # 占用GPU90%的显存 
sess=tf.InteractiveSession(config=config)

vgg_model.load_initial_weights(sess)

with tf.name_scope('train_and_test') as train:
    sess.run(tf.global_variables_initializer())
    
#    ckpt=tf.train.get_checkpoint_state('./checkpoints/')
#    saver.restore(sess, ckpt.model_checkpoint_path)
    
    #将计算图，写到tensorboard
    writer.add_graph(sess.graph)
    
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))
  
    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        acount=0
        acc=0
        print('learn_rate:',learn_rate.eval())
        sess.run(train_iterator)
        for step in range(train_batches_per_epoch):
            if step %20==0:print('batch epoch is ',step)
            img,label=sess.run(next_element)
            
            sess.run(train_op,feed_dict={x:img,y:label})
            acc+=sess.run(accuracy,feed_dict={x:img,y:label})
            acount+=1
            if step%display_step==0:
                s=sess.run(summary_all,feed_dict={x:img,y:label})
                writer.add_summary(s,global_step.eval())
        print('train validation',acc/acount)
        print("{} Start validation".format(datetime.now()))
        sess.run(valid_iterator)
        test_acc=0

        for _ in range(valid_epoch):
            img,label=sess.run(next_element)
            acc=sess.run(accuracy,feed_dict={x:img,y:label})
            test_acc+=acc

            summary = tf.Summary(value=[tf.Summary.Value(tag="v_accuracy", 
                                                    simple_value=acc),])   
    
            writer.add_summary(summary,valid_step.eval())
            sess.run(valid_step.assign_add(1))
        test_acc /= valid_epoch
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))

        print("{} Saving checkpoint of model...".format(datetime.now()))
        
        
        #保存模型
        checkpoint_name=os.path.join(checkpoint_path,
                                     'model_epoch'+str(epoch+1)+'.ckpt')
        save_path=saver.save(sess,checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                      checkpoint_name))
        
        
        
            
            



#for i in range(num_epochs):
