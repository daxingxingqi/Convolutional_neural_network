# Convolutional_neural_network

 *convolutional neural network summary 是一些关于CNN的资料总结。

 *CNN_example 是基于理解总结的一些CNN的应用实例

 *AI_doc 是一个应用实例
 
 *AI_doc 是一个transfer learning应用实例
 
推荐你阅读以下材料来加深对 CNN和Transfer Learning的理解:

>[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

>[Using Convolutional Neural Networks to Classify Dog Breeds](http://cs231n.stanford.edu/reports/2015/pdfs/fcdh_FinalReport.pdf)

>[Building an Image Classifier](https://towardsdatascience.com/learning-about-data-science-building-an-image-classifier-part-2-a7bcc6d5e825)

>[Tips/Tricks in CNN](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)

 1) data augmentation; 
 2) pre-processing on images; 
 3) initializations of Networks; 
 4) some tips during training; 
 5) selections of activation functions; 
 6) diverse regularizations; 
 7) some insights found from figures and finally 
 8) methods of ensemble multiple deep networks.

>[Transfer Learning using Keras](https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8)
1. New dataset is small and similar to original dataset:
There is a problem of over-fitting, if we try to train the entire network. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.

So lets freeze all the VGG19 layers and train only the classifier
```python
for layer in model.layers:
   layer.trainable = False
 
#Now we will be training only the classifiers (FC layers)
```
2. New dataset is large and similar to the original dataset
Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.
``` python
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
3. New dataset is small but very different from the original dataset
Since the dataset is very small, We may want to extract the features from the earlier layer and train a classifier on top of that. This requires a little bit of knowledge on h5py.

The above code should help. It will extract the “block2_pool” features. In general this is not helpful as this layer has (64*64*128) features and training a classifier on top of it might not help us exactly. We can add a few FC layers and train a neural network on top of it. That should be straight forward.

*Add few FC layers and output layer.

*Set the weights for earlier layers and freeze them.

*Train the network.

4. New dataset is large and very different from the original dataset.
This is straight forward. since you have large dataset, you can design your own network or use the existing ones.

*Train the network using random initialisations or use the pre-trained network weights as initialisers. The second one is generally preferred.

*If you are using a different network or making small modification here and there for the existing network, Be careful with the naming conventions.

>[Transfer Learning in TensorFlow on the Kaggle Rainforest competition](https://medium.com/@luckylwk/transfer-learning-in-tensorflow-on-the-kaggle-rainforest-competition-4e978fadb571)

>[Transfer Learning and Fine-tuning](https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2)

相关论文:

>[[VGG16] VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/abs/1409.1556)

>[[Inception-v1] Going deeper with convolutions](https://arxiv.org/abs/1409.4842)

>[[Inception-v3] Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

>[[Inception-v4] Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

>[[ResNet] Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

>[[Xception] Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)



此外，我注意到你使用的图片是从我们提供的数据集中选取的，我非常不推荐这种做法。这种行为可能会导致标签泄露（Label Leakage），并不能很好的评估你的模型的泛化能力。因为，模型在训练过程中本身就是在不断拟合训练集，它能很好地预测训练集里的图片是理所应当的。而验证泛化能力最好的做法就是，使用真实的、在训练集/测试集/验证集都没有出现过的图片来进行测试。你可以自由的使用网上的图片或者自己的图片~😉 同时，希望你能尝试类型的图片来进行实验，比如猫、多条狗（可以是不同品种）、带着狗耳朵的人、风景照等。按照机器学习的思路，你的输入覆盖的输入空间越多，那么你就能对模型进行越好的评估。也就是说，你尝试的图片类型越多，对模型的评估能力就越强。😄

以下是我对改进模型提出的建议，希望对你有帮助：

1.交叉验证（Cross Validation） 在本次训练中，我们只进行了一次训练集/测试集切分，而在实际模型训练过程中，我们往往是使用交叉验证（Cross Validation）来进行模型选择（Model Selection）和调参（Parameter Tunning）的。交叉验证的通常做法是，按照某种方式多次进行训练集/测试集切分，最终取平均值（加权平均值），具体可以参考维基百科)的介绍。

2.模型融合/集成学习（Model Ensembling） 通过利用一些机器学习中模型融合的技术，如voting、bagging、blending以及staking等，可以显著提高模型的准确率与鲁棒性，且几乎没有风险。你可以参考我整理的机器学习笔记中的Ensemble部分。

3.更多的数据 对于深度学习（机器学习）任务来说，更多的数据意味着更为丰富的输入空间，可以带来更好的训练效果。我们可以通过数据增强（Data Augmentation）、对抗生成网络（Generative Adversarial Networks）等方式来对数据集进行扩充，同时这种方式也能提升模型的鲁棒性。

4.更换人脸检测算法 尽管OpenCV工具包非常方便并且高效，Haar级联检测也是一个可以直接使用的强力算法，但是这些算法仍然不能获得很高的准确率，并且需要用户提供正面照片，这带来的一定的不便。所以如果想要获得更好的用户体验和准确率，我们可以尝试一些新的人脸识别算法，如基于深度学习的一些算法。

5.多目标监测 更进一步，我们可以通过一些先进的目标识别算法，如RCNN、Fast-RCNN、Faster-RCNN或Masked-RCNN等，来完成一张照片中同时出现多个目标的检测任务。
