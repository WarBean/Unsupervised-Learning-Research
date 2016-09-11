# 无监督学习论文列表

## 参考来源
- [Most Cited Deep Learning Papers # Unsupervised | Github](https://github.com/terryum/awesome-deep-learning-papers#unsupervised)
- [从OpenAI看深度学习研究前沿 | 知乎专栏](https://zhuanlan.zhihu.com/p/20924929?f3fb8ead20=2fe7890562ecdbf5998ce5a6c0a1ba08)
- [Generative Models | OpenAI博客](https://openai.com/blog/generative-models/)
- [A Path to Unsupervised Learning through Adversarial Networks](https://code.facebook.com/posts/1587249151575490/a-path-to-unsupervised-learning-through-adversarial-networks/)
- [Generative Adversarial Networks（GAN）的现有工作 | 程序媛的日常](http://chuansong.me/n/317902651864)

## 目录
- [生成内容](#生成内容)
- [生成对抗网络Generative Adversarial Network](#生成对抗网络generative-adversarial-network)
- [变分自编码机Variation Auto Encoder](#变分自编码机variation-auto-encoder)
- [Pixel RNN类模型](#pixel-rnn类模型)
- [自编码机Auto Encoder](#自编码机auto-encoder)
- [梯子网络Ladder Network](#梯子网络ladder-network)
- [One-shot Learning](#one-shot-learning)
- [Zero-shot Learning](#zero-shot-learning)
- [Biologically Plausible Learning](#biologically-plausible-learning)
- [其他](#其他)

### 生成内容
[**A Neural Algorithm of Artistic Style**](https://arxiv.org/abs/1508.06576)
- 传说中的Neural Style

[**Image Completion with Deep Learning in TensorFlow**](http://bamos.github.io/2016/08/09/deep-completion/)
（[代码](https://github.com/bamos/dcgan-completion.tensorflow)）
- 用GAN做图像修复（image inpainting任务），主要思想是同时优化两个目标：
- 1.原图中有完好区域和丢失区域，要让生成的修复图与原图在对应的完好区域尽可能接近（所谓Contextual Loss）
- 2.要让生成的修复图尽可能被GAN的判别器判定为真实图片，尽可能像真的（所谓Perceptual Loss）
- 论文：[**Semantic Image Inpainting with Perceptual and Contextual Losses**](https://arxiv.org/abs/1607.07539)

### 生成对抗网络Generative Adversarial Network

[**Generative Adversarial Networks**](http://arxiv.org/abs/1406.2661)
（[代码](https://github.com/goodfeli/adversarial)）
- Goodfellow的GAN开山之作

[**Conditional Generative Adversarial Nets**](https://arxiv.org/abs/1411.1784)

[**Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**](https://arxiv.org/abs/1511.06434)
（[代码](https://github.com/Newmu/dcgan_code)）
- 生成房间图片
- 戴眼镜男人－不戴眼镜男人＋不戴眼镜女人＝戴眼镜女人
- 从一张人脸渐变到另一张人脸

- “这篇论文的提出看似并没有很大创新，但其实它的开源代码现在被使用和借鉴的频率最高……这些工程性的突破无疑是更多人选择 DCGAN 这一工作作为 base 的重要原因”

[**Improved Techniques for Training GANs**](https://arxiv.org/abs/1606.03498)
（[代码](https://github.com/openai/improved-gan)）
- 改变架构，解决GAN训练不稳定的问题
- 半监督学习，少量标注样本，效果比Ladder Network还好一些

[**InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets**](https://arxiv.org/abs/1606.03657)
（[代码](https://github.com/openai/InfoGAN)）
- 对representation code空间施加一些要求，使其更具结构化，而非混沌一团
- 结果在representation向量的单个维度上获得了非常好的可解释性，例如渐变一个维度的数值，生成的人脸图谱从“抬头姿态”到“低头姿态”渐变，非常像流形学习里面的一些例子

### 变分自编码机Variation Auto Encoder

### Pixel RNN类模型
[**Pixel Recurrent Neural Networks**](http://arxiv.org/abs/1601.06759)

[**Conditional Image Generation with PixelCNN Decoders**](http://arxiv.org/abs/1606.05328)

### 自编码机Auto Encoder

[**Stacked What-Where Auto-encoders**](https://arxiv.org/abs/1506.02351)

### 梯子网络Ladder Network

[**From neural PCA to deep unsupervised learning**](https://arxiv.org/abs/1411.7783)
- 提出Ladder架构，但还未做半监督学习

[**Semi-Supervised Learning with Ladder Network**](https://arxiv.org/abs/1507.02672)
- 半监督学习，MNIST用100个标注数据达到约99%，CIFAR用4000个标注数据达到约80%

[**Deconstructing the Ladder Network Architecture**](http://arxiv.org/abs/1511.06430)
- 深入挖掘Ladder Network的原理

### One-shot Learning
[**One-Shot Generalization in Deep Generative Models**](http://arxiv.org/abs/1603.05106)

### Zero-shot Learning

### Biologically Plausible Learning
[**Towards Biologically Plausible Deep Learning**](http://arxiv.org/abs/1502.04156)

[**Towards a Biologically Plausible Backprop**](http://arxiv.org/abs/1602.05179)

[**Feedforward Initialization for Fast Inference of Deep Generative Networks is Biologically Plausible**](http://arxiv.org/abs/1606.01651)


### 其他

[**Towards Principled Unsupervised Learning**](http://arxiv.org/abs/1511.06440)
- 用GAN做半监督学习的论文中所定义的新的损失函数与这篇提出的Output Distribution Matching (ODM) cost有紧密联系
