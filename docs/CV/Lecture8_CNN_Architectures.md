# CNN Architecture

这节课的内容就是构建了一整个卷积神经网络来实现图像分类，并且介绍了 CNN 发展的历史（2012-2019）并且分析了不同的 CNN 架构

## CNN 历史：ImageNet 挑战赛的发展

## LeNet

是第一个真正意义上的卷积神经网络。由于当时的计算能力有限，LeNet 的结构相对简单，训练的数据集也很小。**`它主要应用于邮局的手写数字识别，即MNIST数据集`**。这一限制使得 LeNet 在更复杂的图像任务上应用面较窄，但它为后续 CNN 的发展奠定了基础。

## AlexNet

- **重要性**: AlexNet 是 2012 年由 Alex Krizhevsky 等人提出的，**是深度学习在计算机视觉领域的重要突破**。AlexNet 在 ImageNet 挑战赛中的出色表现，使得深度学习在计算机视觉领域迅速普及，并且标志着大规模深度神经网络在实际应用中的成功。

AlexNet 的架构如下，当初为了在两个 GPU 上计算，使用了两个通道，这也是一种工程技巧，后面随着 GPU 的发展和深度学习框架的出现，我们无需完成那么多的工作量即可轻松复现网络

![](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/AlexNet-Fig_03.png)

![EECS498_L8_28](https://zn-typora-image.oss-cn-hangzhou.aliyuncs.com/typora_image/202408241347838.png)

AlexNet 的第一层是一个卷积层，使用了 11x11 的卷积核，输出 64 个特征通道，步长为 4，填充为 2。这些设计参数是为了在有限的计算能力下尽可能保留图像的特征信息，并减少计算量。

$$
width = height =\dfrac{(227−11+2×2)}{4}+1 =\dfrac{220}{4}+1 = 56
$$

因此第一层的卷积层输出为 $56 \times 56 \times 64$

第二层是一个池化层，Max Pooling，$3 \times 3$，步长为 2

$$
width = height =\dfrac{(56-3)}{2}+1 =\dfrac{53}{2}+1 = 27
$$

Max Pooling 并不改变特征图的通道数，因此第二层池化层的输出为 $27 \times 27 \times 64$

…………

**Flatten 操作**：将池化层输出的 `6×6×256` 张量展平为长度为 `6×6×256 = 9216` 的向量。

**全连接层**: FC1、FC2、FC3，分别输出 4096、4096 和 1000 个特征。

**Softmax 层**: 将最后的 1000 个特征转化为概率分布，输出分类结果。



![EECS498_L8_31](https://zn-typora-image.oss-cn-hangzhou.aliyuncs.com/typora_image/202408241347502.png)

> **内存开销和计算复杂度**

- **卷积层**：卷积层的内存开销与计算复杂度主要由卷积核的数量、大小、输入特征图的大小和通道数决定。卷积操作的本质是计算密集型的，尤其是在输入图像分辨率较高时，早期层需要处理大量的像素点。因此，卷积层在计算过程中会消耗较多的浮点数计算量。
- **全连接层**：全连接层占据了网络中绝大部分的参数。这是因为全连接层将前一层的所有特征图展开成一个长向量，然后与权重矩阵相乘。由于这些向量和权重矩阵通常非常大，导致全连接层的参数数量庞大。例如，AlexNet 中第一个全连接层从 `6×6×256` 的输入生成 4096 个输出，这需要 `6×6×256×4096` 个参数。

> **为什么需要三层全连接层**

**非线性表达能力**：每一层全连接层后面通常跟随一个非线性激活函数（如 ReLU），这大大增加了网络的非线性表达能力。多层全连接层通过不断组合前面层的输出，可以形成更复杂的特征表示。

AlexNet 的 PyTorch 代码实现如下

```python
导入 pytorch 库
import torch
# 导入 torch.nn 模块
from torch import nn
# nn.functional：(一般引入后改名为 F)有各种功能组件的函数实现，如：F.conv2d
import torch.nn.functional as F
 
# 定义 AlexNet 网络模型，继承于父类 nn.Module
class AlexNet(nn.Module):
    # 子类继承中重新定义 Module 类的 __init__()和 forward()函数
    # init()：进行初始化，申明模型中各层的定义
    def __init__(self):
        # super：引入父类的初始化方法给子类进行初始化
        super(AlexNet, self).__init__()
        # 二维卷积层，输入大小为 224 *224，输出大小为 55* 55，输入通道为 3，输出为 96，卷积核为 11，步长为 4
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # 使用 ReLU 作为激活函数，当然也可以使用 Sigmoid 函数等
        self.ReLU = nn.ReLU()
        # MaxPool2d：最大池化操作
        # 二维最大池化层，输入大小为 55 *55，输出大小为 27* 27，输入通道为 96，输出为 96，池化核为 3，步长为 2
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层，输入大小为 27 *27，输出大小为 27* 27，输入通道为 96，输出为 256，卷积核为 5，扩充边缘为 2，步长为 1
        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 最大池化层，输入大小为 27 *27，输出大小为 13* 13，输入通道为 256，输出为 256，池化核为 3，步长为 2
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层，输入大小为 13 *13，输出大小为 13* 13，输入通道为 256，输出为 384，卷积核为 3，扩充边缘为 1，步长为 1
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 卷积层，输入大小为 13 *13，输出大小为 13* 13，输入通道为 384，输出为 384，卷积核为 3，扩充边缘为 1，步长为 1
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 卷积层，输入大小为 13 *13，输出大小为 13* 13，输入通道为 384，输出为 256，卷积核为 3，扩充边缘为 1，步长为 1
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 最大池化层，输入大小为 13 *13，输出大小为 6* 6，输入通道为 256，输出为 256，池化核为 3，步长为 2
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Flatten()：将张量（多维数组）平坦化处理，神经网络中第 0 维表示的是 batch_size，所以 Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()
        # 全连接层
        # Linear（in_features，out_features）
        # in_features 指的是 [batch_size, size] 中的 size, 即样本的大小
        # out_features 指的是 [batch_size，output_size] 中的 output_size，样本输出的维度大小，也代表了该全连接层的神经元个数
        self.f6 = nn.Linear(6*6*256, 4096)
        self.f7 = nn.Linear(4096, 4096)
        # 全连接层&softmax
        self.f8 = nn.Linear(4096, 1000)
        self.f9 = nn.Linear(1000, 2)
 
    # forward()：定义前向传播过程, 描述了各层之间的连接关系
    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.f6(x)
         # Dropout：随机地将输入中 50%的神经元激活设为 0，即去掉了一些神经节点，防止过拟合
        # “失活的”神经元不再进行前向传播并且不参与反向传播，这个技术减少了复杂的神经元之间的相互影响
        x = F.dropout(x, p=0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)
        x = self.f9(x)
        return x
 
# 每个 python 模块（python 文件）都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）
# 如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）
# “__main__” 始终指当前执行模块的名称（包含后缀.py）
# if 确保只有单独运行该模块时，此表达式才成立，才可以进入此判断语法，执行其中的测试代码，反之不行
if __name__ == '__main__':
    # rand：返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数，此处为四维张量
    x = torch.rand([16, 3, 224, 224])
    # 模型实例化
    model = MyAlexNet()
    y = model(x)

```

这是第一个真正意义上的现代神经网络，并且使用了大量的工程技巧，如对图片进行翻转、裁剪，Dropout 正则化等等

![YSAI_ImageClassification_L2_19](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/YSAI_ImageClassification_L2_19.png)

![YSAI_ImageClassification_L2_21](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/YSAI_ImageClassification_L2_21.png)

## ZF Net

![image-20240824141107531](https://zn-typora-image.oss-cn-hangzhou.aliyuncs.com/typora_image/202408241411592.png)

ZFNet (Zeiler and Fergus Net) 是在 AlexNet 基础上优化的网络，通过对超参数的调整提升了性能。主要改进包括：

1. **第一层卷积核与步长调整**：ZFNet 将第一层卷积核尺寸从 AlexNet 的 11x11 缩小到 7x7，同时减小步长为 2（而非 AlexNet 的 4）。这使得网络能够保留更高的空间分辨率，捕捉到更多的细节信息。
2. **总体架构调整**：ZFNet 在保持 AlexNet 的基本结构的同时，优化了卷积层的设置，使得特征提取的精度得到了提升。这表明在当时的计算资源允许范围内，适当增大网络规模可以带来性能提升。

## VGG 网络

VGG 网络通过 **`使用更小的卷积核（如3x3）代替较大的卷积核（如5x5）`** 带来了新的设计理念，核心思想是通过堆叠多个较小的卷积核实现更深的网络，同时在相同计算开销下提高模型的表现。

![image-20240824141347048](https://zn-typora-image.oss-cn-hangzhou.aliyuncs.com/typora_image/202408241413084.png)

**更深更窄的网络效果更佳**：在相同计算成本的情况下，使用多个小卷积核（例如两次 3x3 卷积）可以实现比一次大卷积核（如 5x5 卷积）更好的效果。更深的网络可以捕捉到更复杂的特征。

**规则化设计**：VGG 的设计相比于 AlexNet 更加规则化，所有卷积层的卷积核大小均为 3x3，步幅为 1，填充为 1，池化层均为 2x2 最大池化且步幅为 2。这种统一的设计使网络结构更加简洁、可复用。

**卷积块的使用**：VGG 不是简单地堆叠卷积层，而是通过堆叠多个“卷积块”来构建网络。每个卷积块由几个卷积层和一个池化层组成，这些卷积块堆叠后连接全连接层。这种模块化设计使得 VGG 网络易于扩展和调整。

以下是基于李沐老师的《动手学深度学习》中的 VGG 卷积块代码：

```python
import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
# 示例：构建一个 VGG 块，包含 2 个卷积层，输入通道数为 3，输出通道数为 64
vgg_block_example = vgg_block(2, 3, 64)
print(vgg_block_example)
```



## NiN：网络中的网络

NiN（Network in Network）通过创新性地引入 1x1 卷积核和全局平均池化层，显著提升了卷积神经网络的非线性建模能力和特征表达能力。这种设计在当时引领了卷积神经网络结构的进步，并为后续的深度学习模型提供了新的思路。NiN 的核心思想之一是用 1x1 卷积核替代部分传统卷积层，使得网络不仅在空间维度上进行特征提取，还可以在通道维度上进行更深层次的特征组合。此外，全局平均池化层的引入，有效减少了参数数量和计算复杂度，降低了过拟合风险，同时提供了更紧凑的网络设计。

在实际应用中，NiN 展示了其在图像分类、目标检测和语义分割等任务中的优越性能。其提出的 1x1 卷积核的设计理念也被后续诸多模型广泛采用，如 Inception 系列网络，进一步推动了深度学习的研究和应用。

## GoogleLeNet

GoogleLeNet 是 2014 年谷歌团队为提高神经网络效率、减少复杂性并使其在移动设备上运行而提出的网络模型。它得名于向 Yann LeCun 和他的 LeNet 致敬。GoogleLeNet 的主要贡献之一是引入了 Inception 模块，这种模块结构使网络在保持计算复杂度和参数数量较低的同时，能够显著增加深度。

![EECS498_L8_52](https://zn-typora-image.oss-cn-hangzhou.aliyuncs.com/typora_image/202408241418367.png)

**Inception 模块**：

- Inception 模块是一种复合结构，通过多个不同的卷积核和池化层并行处理输入信息，再在通道维度上合并输出。这种设计避免了传统卷积层的局限性，使网络能够从多个尺度提取特征，增加模型表达能力的同时减少参数量。
- 每个 Inception 模块都包括 1x1、3x3、5x5 的卷积层和 3x3 的最大池化层，并且通过 1x1 卷积进行通道数的压缩以减少计算量。

**网络结构**：

- GoogleLeNet 的深度超过 100 层，是当时深度最大的网络之一。这种深度极大提升了模型的表达能力，尽管当时还没有使用批量归一化（Batch Normalization）技术。

**全局平均池化**：

- 在网络的最后，GoogleLeNet 使用了全局平均池化层来替代全连接层，这减少了模型参数并降低了过拟合风险。

## 批量归一化

批量归一化（Batch Normalization，BN）是一种广泛用于加速神经网络训练收敛的技术。它通过在训练过程中对每个批量数据进行归一化处理，**`使得网络在各层的输入分布更加稳定，有助于解决深度神经网络训练过程中出现的梯度消失和梯度爆炸问题，从而加速收敛`**。

我们知道，损失是在网络最后，当进行反向传播的时候，上流的梯度较大，但是下层的梯度会很小，这是因为每一层的梯度可能都不大，反向传播过程中不断累乘会导致梯度越来越小，也就是说，越靠近数据层，梯度更新的越慢，学习的也就越慢，这就导致了每一层更新下层之后，上层也要重新开始训练，所以批量归一化要解决的问题就是在学习底部层的时候避免变化顶部层

> **批量归一化的原理和作用**

- **均值和方差的归一化**：
批量归一化的核心思想是对每个批量的数据进行均值和方差的归一化处理，使得每个特征的 **`分布稳定在0均值和单位方差附近`**。这样可以减少不同批次数据分布带来的影响，有利于提高网络的泛化能力和加快训练速度。

- **归一化公式**：

对于一个批量数据集合 $B$，批量归一化的数学表达式如下： 

$$
\mu_B = \frac{1}{|B|}\sum_{i\in B}x_i\\
\sigma_B^2 = \frac{1}{|B|}\sum_{i\in B}(x_i - \mu_B)^2 + \epsilon
$$
    
其中，$\mu_B$ 是批量数据的均值，$\sigma_B^2$ 是批量数据的方差（加上小的常数 $\epsilon$ 来防止除以零），$x_i$ 是批量中的每个样本。
  
- **归一化后的变换**：

归一化后，每个样本 $x_i$ 被转换为：

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
    
然后通过可学习的拉伸参数 $\gamma$ 和偏移参数 $\beta$ 来调整归一化后的结果： 

$$
y_i = \gamma \hat{x}_i + \beta
$$

这里 $\gamma$ 和 $\beta$ 是可以通过反向传播进行优化的参数，用来学习最佳的数据分布。
  
- **作用于全连接层和卷积层**：

  批量归一化通常 **`应用在每层的激活函数之前，即作用于全连接层和卷积层的输出`**。这种方式有助于使得每层输入的分布更稳定，提高了网络训练时的梯度流动性，加速了模型的收敛速度。





## ResNet

ResNet（Residual Network，残差网络），它通过引入 **残差学习** 的思想，**成功解决了深层神经网络训练中的梯度消失问题**，使得网络深度能够大幅度增加，推动了深度学习的发展。

**残差学习（Residual Learning）**：

- ResNet 的核心思想是引入“**残差块**”（Residual Block），通过 **跳跃连接（skip connection）直接将输入信息传递到更深的层，这样可以避免网络在层数增加时出现的退化问题。** 残差块的结构如下：$y = F(x, \{W_i\}) + x$
- 其中，$x$ 是输入，$F(x, {W_i})$ 是卷积层的输出，残差块直接将输入 $x$ 加到输出上。这样，即使 $F(x)$ 趋近于 0，网络也可以通过跳跃连接保持信息的流动。

**解决梯度消失问题**：

- 随着网络深度的增加，梯度消失和梯度爆炸问题会变得更加严重，导致训练困难。通过引入残差学习，ResNet 可以确保梯度能够顺畅地从网络后层传播到前层，从而缓解了梯度消失的问题，**使得网络可以轻松地增加到数百层深**。

**深层网络的优势**：

- 在 ResNet 之前，网络深度的增加并不总是带来更好的性能，甚至可能导致模型退化，出现训练误差增大的现象。ResNet 通过残差块使得非常深的网络也能够有效训练，从而显著提高了模型的表现。在 ImageNet 挑战赛中，ResNet-152 显著超越了之前的网络结构，获得了冠军。

## MobileNet

> **MobileNet 的优势**

- **高效性**：通过深度可分离卷积、宽度乘子和分辨率乘子的结合，MobileNet 大幅度减少了模型的计算量和参数量，使其在资源受限的设备上能够高效运行。
- **可定制性**：用户可以根据计算资源和应用需求，通过调整宽度乘子和分辨率乘子来灵活控制模型的复杂度，达到准确率与效率的平衡。
- **广泛应用**：MobileNet 广泛应用于各种移动和嵌入式设备上的任务，如图像分类、目标检测、人脸识别等。

> **MobileNet 的局限性**

尽管 MobileNet 在轻量化设计上取得了成功，但由于其计算量和参数量的减少，其在某些任务上的准确率可能不如更复杂的大型网络。为了应对这一挑战，后续提出了 MobileNetV2 和 MobileNetV3，这些改进版本进一步优化了网络结构，提升了准确率和性能。

