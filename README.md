# finetune-vgg-16
本文以实战为主，主要介绍基于tensorflow搭建vgg-16网络，vgg设计思想原理，在此不做多余的赘述。 
vgg-16的结构单独封装到一个vgg16_structure.py文件里面，以[weights，biases]的结构封装

structure = {

    # convolution layer 1
    'conv1_1': [[3, 3, 3, 64], [64]],
    'conv1_2': [[3, 3, 64, 64], [64]],

    # convolution layer 2
    'conv2_1': [[3, 3, 64, 128], [128]],
    'conv2_2': [[3, 3, 128, 128], [128]],

    # convolution layer 3
    'conv3_1': [[3, 3, 128, 256], [256]],
    'conv3_2': [[3, 3, 256, 256], [256]], 
    'conv3_3': [[3, 3, 256, 256], [256]],

    # convolution layer 4
    'conv4_1': [[3, 3, 256, 512], [512]],
    'conv4_2': [[3, 3, 512, 512], [512]],
    'conv4_3': [[3, 3, 512, 512], [512]],

    # convolution layer 5
    'conv5_1': [[3, 3, 512, 512], [512]],
    'conv5_2': [[3, 3, 512, 512], [512]],
    'conv5_3': [[3, 3, 512, 512], [512]],

    # fully-connection 6
    'fc6': [[4096, 0, 0, 0], [4096]],

    # fully-connection 7
    'fc7': [[4096, 0, 0, 0], [4096]],

    # fully-connection 8
    'fc8': [[1000, 0, 0, 0], [1000]],

}
为了快速的搭建卷积网络，这里封装一个卷积操作函数

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

最后两层是全连接，所以在封装一个全连接层

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

这里不得不提的是reduce操作，完美的使这个函数可以复用到任何网络中去

             shape=X.get_shape().as_list()
             print(name,shape)
             dim=reduce(lambda x,y:x*y,shape[1:])

运算过程如下如果shape为[3,4,5,3] 也就是batch_size=3 ,长4宽5，通道数为3，如果转换为一维度 很显然为3*4*5*3=180 
使用reduce函数很容易达到这个效果 
在vgg中引用inception-v2中的BN正则化方法，BN目的是使得每层训练的输出结果在同一分布下，实验证明不仅可以加速收敛速度，还可以提高准确度，具体实现如下

def batch_normalization(self,input, decay=0.9, eps=1e-5):
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

tf.nn.moments

该函数是计算矩阵的方差与均值，因为如果想要计算所有图像的均值与方差，显然不太现实，所以每次计算每个batch的方差与均值，为了使得每个batch的方差与均值尽可能的接近整体分布方差与均值的估计值，这里采用一种指数移动平均，平滑操作

ema_apply_op = tf.train.ExponentialMovingAverage(decay=decay)

你需要设置衰减系数 
该op有两个方法，apply与average

ema_apply_op = ema.apply([batch_mean, batch_var])

apply方法，会返回一个经过指数移动平均平滑后的数，但是在测试中，我们并不希望继续做平滑操作，所以可以使用average方法

 with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

这是一种依赖写法，表示想要得到返回值，就得运行ema_apply_op操作，然而我们希望能够保存均值与方差，希望能在测试加载模型的时候得到它，那么tf.identity方法，会产生一个新的op保存在计算图中，以便下次加载模型的时候能够直接得到

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

做迁移学习，自然需要使用已经训练好的参数，vgg参数我已经保存到.npy文件中，可以直接加载到运算图中，此段代码，需要引起你注意的是

with tf.variable_scope(op_name,reuse=True)

设置变量可复用，为了防止篇幅过大进一步的解释不再继续

数据集自然是要使用tensorflow的tfrecords，高效方便，建议tensorflow版本高于1.4，因为处理文件输入流的方式我没有使用繁琐的队列，而是使用简单易用的Dataset,在1.4版本之后，归为核心方法，放到 tf.data.Dataset 
并且支持链式操作，很方便

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

dataset的迭代器，有One-shot Iterator、Initializable Iterator、Reinitializable Iterator、Feedable Iterator四种，为了方便使用训练集与测试集，我使用了Reinitializable Iterator

train_dataset=load_dataset("train/train.tfrecords").batch(batch_size)
valid_dataset=load_dataset("valid/valid.tfrecords").batch(batch_size)

iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
train_iterator=iterator.make_initializer(train_dataset,name='trian_iterator')
valid_iterator=iterator.make_initializer(valid_dataset,name='trian_iterator')
next_element=iterator.get_next()

设置GPU操作，我的是英伟达泰坦 内存13G,图像大小为227*227，batchsize不要设置过大，否则容易内存溢出，我设置的100内存溢出，最终设置为30，占用GPU资源设置为按需加载

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#设置按需增长
config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True

sess=tf.InteractiveSession(config=config)

vgg_model.load_initial_weights(sess)

使用该模型，你需要设置你的tfrecord文件位置，batch_size大小，与vgg网络结构全链接层的最后一层，神经元个数改为你的类别个数 
13类，3轮跑到53%,还不错 
博客地址：https://blog.csdn.net/u014296502/article/details/80367140
