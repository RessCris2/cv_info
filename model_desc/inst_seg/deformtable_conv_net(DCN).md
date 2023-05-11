# Deformable Convolutional Networks
## 参考
- [Deformable Convolution](https://zhuanlan.zhihu.com/p/138020203)
## 提出背景
CNN 这个架构本身是受形状限制的。（形状？）

Deformable Convolution Networks是MSRA的代季锋和一帮实习生在2017年搞出的一种全新的卷积结构。这种方法将固定形状的卷积过程改造成了能适应物体形状的可变的卷积过程，从而使结构适应物体形变的能力更强。新的结构在PASCAL VOC和COCO数据集上都表现出了不错的成绩。  

## 具体方法
- deformable convolution 
- deformable RoI pooling

> augmenting the spatial sampling locations in the modules with additional offsets and learning the offsets from target tasks, without additional supervision.

相当于是一个改进 CNN 的架构。

## 为什么可以解决这个问题？


## 效果？


## 参考代码
