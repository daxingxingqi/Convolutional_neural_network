# 1.Transfer Learning 

推荐你阅读以下材料来加深对 CNN和Transfer Learning的理解:

**[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)**

> 点这里查看[笔记](https://github.com/daxingxingqi/CS231n-2017-Summary)

> 点这里查看官方[笔记](https://cs231n.github.io/)

**[Using Convolutional Neural Networks to Classify Dog Breeds](http://cs231n.stanford.edu/reports/2015/pdfs/fcdh_FinalReport.pdf)**

**[Building an Image Classifier](https://towardsdatascience.com/learning-about-data-science-building-an-image-classifier-part-2-a7bcc6d5e825)**

**[Tips/Tricks in CNN](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)**

- 1) data augmentation; 
- 2) pre-processing on images; 
- 3) initializations of Networks; 
- 4) some tips during training; 
- 5) selections of activation functions; 
- 6) diverse regularizations; 
- 7) some insights found from figures and finally 
- 8) methods of ensemble multiple deep networks.

## [Transfer Learning using Keras](https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8)

> 1. **New dataset is small and similar to original dataset:**
>    There is a problem of over-fitting, if we try to train the entire network. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.

So lets freeze all the VGG19 layers and train only the classifier

```python
for layer in model.layers:
   layer.trainable = False
 
#Now we will be training only the classifiers (FC layers)
```

> 1. **New dataset is large and similar to the original dataset**
>    Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.

```python
for layer in model.layers:
   layer.trainable = True
#The default is already set to True. I have mentioned it here to make things clear.
```

In case if you want to freeze the first few layers as these layers will be detecting edges and blobs, you can freeze them by using the following code.

```python
for layer in model.layers[:5]:
   layer.trainable = False.
# Here I am freezing the first 5 layers 
```

> 1. **New dataset is small but very different from the original dataset**
>    Since the dataset is very small, We may want to extract the features from the earlier layer and train a classifier on top of that. This requires a little bit of knowledge on h5py.The above code should help. It will extract the “block2_pool” features. In general this is not helpful as this layer has (64*64*128) features and training a classifier on top of it might not help us exactly. We can add a few FC layers and train a neural network on top of it. That should be straight forward.

> - Add few FC layers and output layer.

> - Set the weights for earlier layers and freeze them.

> - Train the network.

> 1. **New dataset is large and very different from the original dataset.**
>    This is straight forward. since you have large dataset, you can design your own network or use the existing ones.

> - Train the network using random initialisations or use the pre-trained network weights as initialisers. The second one is generally preferred.

> - If you are using a different network or making small modification here and there for the existing network, Be careful with the naming conventions.

[Transfer Learning in TensorFlow on the Kaggle Rainforest competition](https://medium.com/@luckylwk/transfer-learning-in-tensorflow-on-the-kaggle-rainforest-competition-4e978fadb571)

[Transfer Learning and Fine-tuning](https://medium.com/deeplearningsandbox/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2)

# 2.框架论文

**[[VGG16] VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/abs/1409.1556)**

- VGG16

```python
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

input_shape = (224, 224, 3)

model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(1000, activation='softmax')
])

model.summary()
```

- VGG19

```python
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

input_shape = (224, 224, 3)

model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',)
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',)
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',)
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(1000, activation='softmax')
])

model.summary()
```

**[[Inception-v1] Going deeper with convolutions](https://arxiv.org/abs/1409.4842)**

为什么1*1卷积-https://iamaaditya.github.io/2016/03/one-by-one-convolution/

讲解-https://becominghuman.ai/understanding-and-coding-inception-module-in-keras-eb56e9056b4b

<div align=center><img width="550" src=resource/1.png></div>

```python
#-*- coding: UTF-8 -*-
"""
Author: lanbing510
Environment: Keras2.0.5，Python2.7
Model: GoogLeNet Inception V1
"""

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input, concatenate
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
from KerasLayers.Custom_layers import LRN2D


# Global Constants
NB_CLASS=1000
LEARNING_RATE=0.01
MOMENTUM=0.9
ALPHA=0.0001
BETA=0.75
GAMMA=0.1
DROPOUT=0.4
WEIGHT_DECAY=0.0005
LRN2D_NORM=True
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
USE_BN=True


def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=WEIGHT_DECAY):
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None

    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    if lrn2d_norm:
        x=LRN2D(alpha=ALPHA,beta=BETA)(x)

    return x



def inception_module(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=None):
    (branch1,branch2,branch3,branch4)=params
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None

    pathway1=Conv2D(filters=branch1[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    pathway2=Conv2D(filters=branch2[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway2=Conv2D(filters=branch2[1],kernel_size=(3,3),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway2)

    pathway3=Conv2D(filters=branch3[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway3=Conv2D(filters=branch3[1],kernel_size=(5,5),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway3)

    pathway4=MaxPooling2D(pool_size=(3,3),strides=1,padding=padding,data_format=DATA_FORMAT)(x)
    pathway4=Conv2D(filters=branch4[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway4)

    return concatenate([pathway1,pathway2,pathway3,pathway4],axis=concat_axis)



def create_model():
    if DATA_FORMAT=='channels_first':
        INP_SHAPE=(3,224,224)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=1
    elif DATA_FORMAT=='channels_last':
        INP_SHAPE=(224,224,3)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=3
    else:
        raise Exception('Invalid Dim Ordering: '+str(DIM_ORDERING))

    x=conv2D_lrn2d(img_input,64,(7,7),2,padding='same',lrn2d_norm=False)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    x=LRN2D(alpha=ALPHA,beta=BETA)(x)

    x=conv2D_lrn2d(x,64,(1,1),1,padding='same',lrn2d_norm=False)

    x=conv2D_lrn2d(x,192,(3,3),1,padding='same',lrn2d_norm=True)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(64,),(96,128),(16,32),(32,)],concat_axis=CONCAT_AXIS) #3a
    x=inception_module(x,params=[(128,),(128,192),(32,96),(64,)],concat_axis=CONCAT_AXIS) #3b
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(192,),(96,208),(16,48),(64,)],concat_axis=CONCAT_AXIS) #4a
    x=inception_module(x,params=[(160,),(112,224),(24,64),(64,)],concat_axis=CONCAT_AXIS) #4b
    x=inception_module(x,params=[(128,),(128,256),(24,64),(64,)],concat_axis=CONCAT_AXIS) #4c
    x=inception_module(x,params=[(112,),(144,288),(32,64),(64,)],concat_axis=CONCAT_AXIS) #4d
    x=inception_module(x,params=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS) #4e
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS) #5a
    x=inception_module(x,params=[(384,),(192,384),(48,128),(128,)],concat_axis=CONCAT_AXIS) #5b
    x=AveragePooling2D(pool_size=(7,7),strides=1,padding='valid',data_format=DATA_FORMAT)(x)

    x=Flatten()(x)
    x=Dropout(DROPOUT)(x)
    x=Dense(output_dim=NB_CLASS,activation='linear')(x)
    x=Dense(output_dim=NB_CLASS,activation='softmax')(x)

    return x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT


def check_print():
    # Create the Model
    x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT=create_model()

    # Create a Keras Model
    model=Model(input=img_input,output=[x])
    model.summary()

    # Save a PNG of the Model Build
    plot_model(model,to_file='GoogLeNet.png')

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
    print 'Model Compiled'


if __name__=='__main__':
    check_print() 
```

 **[[Inception V2], Batch Normalization:Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)**

Inception V2的网络在Inception v1的基础上，进行了改进，一方面了加入了BN层，减少了Internal Covariate Shift（内部神经元分布的改变），使每一层的输出都规范化到一个N(0, 1)的高斯，还去除了Dropout、LRN等结构；另外一方面学习VGG用2个3x3的卷积替代inception模块中的5x5卷积，既降低了参数数量，又加速计算。

**[[Inception-v3] Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)**

 Inception V3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1）。这样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性，可以处理更多更丰富的空间特征，增加特征多样性。还有值得注意的地方是网络输入从224x224变为了299x299，更加精细设计了35x35/17x17/8x8的模块。

**[[Inception-v4] Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)**

Inception V4结合了微软的ResNet，发现ResNet的结构可以极大地加速训练，同时性能也有提升，得到一个Inception-ResNet V2网络，同时还设计了一个更深更优化的Inception V4模型，能达到与Inception-ResNet V2相媲美的性能。

**[[ResNet] Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)**

讲解-https://blog.waya.ai/deep-residual-learning-9610bb62c355

**[[Xception] Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)**

讲解-https://blog.csdn.net/u014380165/article/details/75142710

# 损失和优化算法

*损失函数是用来估量模型中预测值y与真实值Y之间的差异，即不一致程度

如果你想详细了解 Keras 中的完全连接层，请阅读这篇关于密集层的[文档](https://keras.io/layers/core/)。你可以通过为 **kernel_initializer** 和 **bias_initializer** 参数提供值更改权重的初始化方法。注意默认值分别为 **'glorot_uniform'** 和 **'zeros'**。你可以在相应的 Keras [文档](https://keras.io/initializers/)中详细了解每种初始化程序的工作方法。

Keras 中有很多不同的[损失函数](https://keras.io/losses/)。对于这节课来说，我们将仅使用 **categorical_crossentropy**。

参阅 Keras 中可[用的优化程序列表](https://keras.io/optimizers/)。当你编译模型（在记事本的第 7 步）时就会指定优化程序。

> **'sgd'** : SGD

> **'rmsprop'** : RMSprop

> **'adagrad'** : Adagrad

> **'adadelta'** : Adadelta

> **'adam'** : Adam

> **'adamax'** : Adamax

> **'nadam'** : Nadam

> **'tfoptimizer'** : TFOptimizer

**关于激活函数的[文档](http://cs231n.github.io/neural-networks-1/#actfun)**

# checkpoint

*在训练过程中，你可以使用很多回调（例如 ModelCheckpoint）来监控你的模型。你可以参阅此处的[详情内容](https://keras.io/callbacks/#modelcheckpoint)。建议你先详细了解 EarlyStopping 回调。如果你想查看另一个 ModelCheckpoint 代码示例，请参阅[这篇博文](http://machinelearningmastery.com/check-point-deep-learning-models-keras/)。

*Mnist 参阅[其他分类器](http://yann.lecun.com/exdb/mnist/)的效果

# 池化

请参阅该 Keras [文档](https://keras.io/layers/pooling/)，了解不同类型的池化层！

论文[network in network](https://arxiv.org/abs/1312.4400)

参阅 CIFAR-10 竞赛的[获胜架构](http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/)！

# 数据增强

关于 `steps_per_epoch` 的注意事项

`fit_generator` 具有很多参数，包括

```python
steps_per_epoch = x_train.shape[0] / batch_size
```

其中 `x_train.shape[0]` 对应的是训练数据集 x_train 中的独特样本数量。通过将 steps_per_epoch 设为此值，我们确保模型在每个 epoch 中看到 `x_train.shape[0]` 个增强图片。

> 阅读这篇对 MNIST 数据集进行可视化的[精彩博文](http://machinelearningmastery.com/image-augmentation-deep-learning-keras/)。

> 参阅此[详细实现](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)，了解如何使用增强功能提高 Kaggle 数据集的效果。

> 阅读关于 ImageDataGenerator 类的 Keras [文档](https://keras.io/preprocessing/image/)。

# 防止过拟合

# 补充资料

参阅 [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 论文！

在此处详细了解 [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)。

此处是 [ResNet](https://arxiv.org/pdf/1512.03385v1.pdf) 论文。

这是用于访问一些著名 CNN 架构的 Keras [文档](https://keras.io/applications/)。

阅读这一关于梯度消失问题的[详细处理方案](http://neuralnetworksanddeeplearning.com/chap5.html)。

这是包含不同 CNN 架构的基准的 GitHub [资源库](https://github.com/jcjohnson/cnn-benchmarks)。

访问 [ImageNet Large Scale Visual Recognition Competition (ILSVRC)](http://www.image-net.org/challenges/LSVRC/) 网站。

可以在[此处](https://github.com/udacity/machine-learning/tree/master/projects/practice_projects/cnn)链接的 GitHub 资源库中访问视频中提到的 Jupyter Notebook。转到 transfer-learning/ 文件夹并打开 transfer_learning.ipynb。如果你想了解如何计算自己的瓶颈特征，请查看 bottleneck_features.ipynb（你可能无法在 AWS GPU 实例上运行 bottleneck_features.ipynb，如果是这种情况，你可以在本地 CPU/GPU 上使用 notebook！）

课外资料
这是提议将 GAP 层级用于对象定位的[首篇研究论文](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)。
参阅这个使用 CNN 进行对象定位的[资源库](https://github.com/alexisbcook/ResNetCAM-keras)。
观看这个关于使用 CNN 进行对象定位的[视频演示](https://www.youtube.com/watch?v=fZvOy0VXWAI)(Youtube链接，国内网络可能打不开)。
参阅这个使用可视化机器更好地理解瓶颈特征的[资源库](https://github.com/alexisbcook/keras_transfer_cifar10)。

（非常棒的）课外资料 ！

注：由于以下部分链接来自于外网，国内网络可能打不开

如果你想详细了解如何解读 CNN（尤其是卷积层），建议查看以下资料：

> 这是摘自斯坦福大学的 CS231n 课程中的一个a [章节](http://cs231n.github.io/understanding-cnn/)，其中对 CNN 学习的内容进行了可视化。

> 参阅这个关于很酷的 [OpenFrameworks](http://openframeworks.cc/) 应用的[演示](https://aiexperiments.withgoogle.com/what-neural-nets-see)，该应用可以根据用户提供的视频实时可视化 CNN！

> 这是另一个 CNN 可视化工具的[演示](https://www.youtube.com/watch?v=AgkfIQ4IGaM&t=78s)。如果你想详细了解这些可视化图表是如何制作的，请观看此[视频](https://www.youtube.com/watch?v=ghEmQSxT6tw&t=5s)。

> 这是另一个可与 Keras 和 Tensorflow 中的 CNN 无缝合作的[可视化工具](https://medium.com/merantix/picasso-a-free-open-source-visualizer-for-cnns-d8ed3a35cfc5)。

> 阅读这篇可视化 CNN 如何看待这个世界的 [Keras 博文](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)。在此博文中，你会找到 Deep Dreams 的简单介绍，以及在 Keras 中自己编写 Deep Dreams 的代码。阅读了这篇博文后：

> 再观看这个利用 [Deep Dreams](https://www.youtube.com/watch?v=XatXy6ZhKZw) 的音乐视频（注意 3:15-3:40 部分）！

> 使用这个[网站](https://deepdreamgenerator.com/)创建自己的 Deep Dreams（不用编写任何代码！）。

如果你想详细了解 CNN 的解释

> 这篇[文章](https://blog.openai.com/adversarial-example-research/)详细讲解了在现实生活中使用深度学习模型（暂时无法解释）的一些危险性。

> 这一领域有很多热点研究。[这些作者](https://arxiv.org/abs/1611.03530)最近朝着正确的方向迈出了一步。

# CNN 应用案例

课外资料
注：部分资料来自国外 youtube 与 google research.

**了解 [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) 模型。**

> 如果你能训练人工智能机器人唱歌，干嘛还训练它聊天？在 2017 年 4 月，研究人员使用 WaveNet 模型的变体生成了歌曲。原始论文和演示可以在[此处](http://www.creativeai.net/posts/W2C3baXvf2yJSLbY6/a-neural-parametric-singing-synthesizer)找到。

**了解[文本分类CNN](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) 。**

> 你或许想注册作者的深度学习简讯！

**了解 Facebook 的[创新 CNN 方法(Facebook)](https://code.facebook.com/posts/1978007565818999/a-novel-approach-to-neural-machine-translation/)**，该方法专门用于解决语言翻译任务，准确率达到了前沿性水平，并且速度是 RNN 模型的 9 倍。

**利用 CNN 和强化学习玩 [Atari](https://deepmind.com/research/dqn/) 游戏。你可以[下载](https://sites.google.com/a/deepmind.com/dqn/)此论文附带的代码。**

> 如果你想研究一些（深度强化学习）初学者代码，建议你参阅 [Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/) 的帖子。

**利用 CNN [玩看图说词游戏](https://quickdraw.withgoogle.com/#)！**

> 此外，还可以参阅 [A.I. Experiments](https://aiexperiments.withgoogle.com/) 网站上的所有其他很酷的实现。别忘了 [AutoDraw](https://www.autodraw.com/)！

**详细了解 [AlphaGo](https://deepmind.com/research/alphago/)。**

> 阅读[这篇文章](https://www.technologyreview.com/s/604273/finding-solace-in-defeat-by-artificial-intelligence/?set=604287)，其中提出了一个问题：如果掌控 Go“需要人类直觉”，那么人性受到挑战是什么感觉？_

**观看这些非常酷的视频，其中的无人机都受到 CNN 的支持。**

> 这是初创企业 [Intelligent Flying Machines (IFM)](https://www.youtube.com/watch?v=AMDiR61f86Y) (Youtube)的访谈。

> 户外自主导航通常都要借助[全球定位系统 (GPS)](http://www.droneomega.com/gps-drone-navigation-works/)，但是下面的演示展示的是由 CNN 提供技术支持的[自主无人机](https://www.youtube.com/watch?v=wSFYOw4VIYY)(Youtube)。

**如果你对无人驾驶汽车使用的 CNN 感兴趣，请参阅：**

> 我们的[无人驾驶汽车工程师纳米学位课程](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)，我们在[此项目](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)中对[德国交通标志](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)数据集中的标志进行分类。

> 这些[系列博客](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/)，其中详细讲述了如何训练用 Python 编写的 CNN，以便生成能够玩“侠盗猎车手”的无人驾驶 AI。

**参阅视频中没有提到的其他应用情形。**

> 一些全球最著名的画作被[转换成了三维形式](http://www.businessinsider.com/3d-printed-works-of-art-for-the-blind-2016-1)，以便视力受损人士也能欣赏。虽然这篇文章没有提到是怎么做到的，我们注意到可以使用 CNN [预测单个图片的深度](https://www.cs.nyu.edu/~deigen/depth/)。

> 参阅这篇关于使用 CNN 确定乳腺癌位置的[研究论文](https://research.googleblog.com/2017/03/assisting-pathologists-in-detecting.html)(google research)。

> CNN 被用来[拯救濒危物种](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)！

> 一款叫做 [FaceApp](http://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/) 的应用使用 CNN 让你在照片中是微笑状态或改变性别。

此外，我注意到你使用的图片是从我们提供的数据集中选取的，我非常不推荐这种做法。这种行为可能会导致标签泄露（Label Leakage），并不能很好的评估你的模型的泛化能力。因为，模型在训练过程中本身就是在不断拟合训练集，它能很好地预测训练集里的图片是理所应当的。而验证泛化能力最好的做法就是，使用真实的、在训练集/测试集/验证集都没有出现过的图片来进行测试。你可以自由的使用网上的图片或者自己的图片~😉 同时，希望你能尝试类型的图片来进行实验，比如猫、多条狗（可以是不同品种）、带着狗耳朵的人、风景照等。按照机器学习的思路，你的输入覆盖的输入空间越多，那么你就能对模型进行越好的评估。也就是说，你尝试的图片类型越多，对模型的评估能力就越强。😄

以下是我对改进模型提出的建议，希望对你有帮助：

1.交叉验证（Cross Validation） 在本次训练中，我们只进行了一次训练集/测试集切分，而在实际模型训练过程中，我们往往是使用交叉验证（Cross Validation）来进行模型选择（Model Selection）和调参（Parameter Tunning）的。交叉验证的通常做法是，按照某种方式多次进行训练集/测试集切分，最终取平均值（加权平均值），具体可以参考维基百科)的介绍。

2.模型融合/集成学习（Model Ensembling） 通过利用一些机器学习中模型融合的技术，如voting、bagging、blending以及staking等，可以显著提高模型的准确率与鲁棒性，且几乎没有风险。你可以参考我整理的机器学习笔记中的Ensemble部分。

3.更多的数据 对于深度学习（机器学习）任务来说，更多的数据意味着更为丰富的输入空间，可以带来更好的训练效果。我们可以通过数据增强（Data Augmentation）、对抗生成网络（Generative Adversarial Networks）等方式来对数据集进行扩充，同时这种方式也能提升模型的鲁棒性。

4.更换人脸检测算法 尽管OpenCV工具包非常方便并且高效，Haar级联检测也是一个可以直接使用的强力算法，但是这些算法仍然不能获得很高的准确率，并且需要用户提供正面照片，这带来的一定的不便。所以如果想要获得更好的用户体验和准确率，我们可以尝试一些新的人脸识别算法，如基于深度学习的一些算法。

5.多目标监测 更进一步，我们可以通过一些先进的目标识别算法，如RCNN、Fast-RCNN、Faster-RCNN或Masked-RCNN等，来完成一张照片中同时出现多个目标的检测任务。

