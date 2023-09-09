# Dataset distillation的学习

## 0 相关文章

[1]  Dataset distillation

[2]  Dataset Distillation using Neural Feature Regression

[3]  Data Distillation Towards Omni-Supervised Learning (CVPR2018)

[4]  Dataset Distillation by Matching Training Trajectories (CVPRW2022)

[5]  FedMD-Heterogenous Federated Learning

[6]  Flexible Dataset Distillation

[7]  Learning From Noisy (ICCV2017)

[8]  Dataset Distillation with Infinitely Wide Convolutional Networks (NeurIPS-2021)

[9]  Soft-Label Dataset Distillation and Text Dataset Distillation

[10]Wearable ImageNet Synthesizing Tileable Textures via Dataset Distillation (CVPRW 2022)

## 1 文章简读

 **[1]  Dataset distillation**

合成少量数据，这些数据不需要一定来自正确的数据分布，但是当作为模型的训练数据学习时，能达到近似在原始数据上训练的效果。

 **[2]  Dataset Distillation using Neural Feature Regression**

 

 **[3]  Data Distillation Towards Omni-Supervised Learning (CVPR2018)**

 **[4]  Dataset Distillation by Matching Training Trajectories (CVPRW2022)**

 **[5]  FedMD-Heterogenous Federated Learning**

 **[6]  Flexible Dataset Distillation**

 **[7]  Learning From Noisy (ICCV2017)**

 **[8]  Dataset Distillation with Infinitely Wide Convolutional Networks (NeurIPS-2021)**

 **[9]  Soft-Label Dataset Distillation and Text Dataset Distillation**

 **[10]Wearable ImageNet Synthesizing Tileable Textures via Dataset Distillation (CVPRW 2022)**

## 2 具体实现

### [1]  Dataset distillation

Consider a training dataset : $X = {x_i}^N_{i=1}$  <!--训练数据集-->

parameterize the neural network as θ <!--模型-->

denote $l(x_i,\theta)$ as the loss function that represents the loss of this network on a data point $x_i$.

 这篇文章的任务是最小化误差 $\theta ^* $:

<img src="C:\Users\kiko\AppData\Roaming\Typora\typora-user-images\image-20220831134216416.png" alt="image-20220831134216416" style="zoom:50%;" />

<!--这不就是经典的loss？不明所以的文章-->

**OPTIMIZING DISTILLED DATA：**

对于minibatch数据$X_t = {x_{t,j}}^n_{j=1}$有以下更新公式：

<img src="C:\Users\kiko\AppData\Roaming\Typora\typora-user-images\image-20220831142730881.png" alt="image-20220831142730881" style="zoom:67%;" />

本文提出一个数据集$ \tilde X$ 远小于数据集$ X $,权重的更新如下：

<img src="C:\Users\kiko\AppData\Roaming\Typora\typora-user-images\image-20220831143749766.png" alt="image-20220831143749766" style="zoom:67%;" />

提出一个初始化的参数（权重）$ \theta _0$，用上面的式子得到一个$\theta _1$，

通过最小化下式的  $\mathcal{L}$，就能得到$\tilde X$和$\tilde \eta$

![image-20220831143805895](C:\Users\kiko\AppData\Roaming\Typora\typora-user-images\image-20220831143805895.png)

在上式中，利用$\theta _1$作为$\tilde X$和$\tilde \eta$的函数，计算所有训练数据上的新的权重。loss $\mathcal L$是关于$\tilde X$和$\tilde \eta$可微的，

**DISTILLATION FOR RANDOM INITIALIZATIONS**：

上面为给定初始化优化的提取数据不能很好地推广到其他初始化。蒸馏数据通常看起来像随机噪声，如下图：

![image-20220831152759044](C:\Users\kiko\AppData\Roaming\Typora\typora-user-images\image-20220831152759044.png)

因为它对训练数据集x和特定网络初始化$\theta _0$的信息进行了编码。

为了解决这个问题，我们转而计算少量提取的数据，这些数据可以用于具有来自特定分布的随机初始化的网络。我们将优化问题表述为:

<img src="C:\Users\kiko\AppData\Roaming\Typora\typora-user-images\image-20220831153017061.png" alt="image-20220831153017061" style="zoom: 50%;" />

其中网络初始化$\theta _0$是从分布$p(\theta _0)$中随机采样的。算法1说明了我们的主要方法。

为了蒸馏数据更好地被学习，$\mathscr l(X ,\cdot)$ 共享相似的局部状态（如输出的值、梯度大小）



