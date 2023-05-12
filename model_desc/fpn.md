# Feature Pyramid Network


[【目标检测】FPN(Feature Pyramid Network)](https://zhuanlan.zhihu.com/p/62604038) 写得非常好，非常细致。
## 提出背景
FPN 是为了解决大小不同的目标检测问题的。

为了使得不同尺度的特征都包含丰富的语义信息，同时又不使得计算成本过高，作者就采用top down和lateral connection的方式，让低层高分辨率低语义的特征和高层低分辨率高语义的特征融合在一起，使得最终得到的不同尺度的特征图都有丰富的语义信息

## 具体实现
![fpn](https://github.com/RessCris2/cv_info/blob/main/imgs/fpn_1.jpg)

特征金字塔的结构主要包括三个部分：bottom-up，top-down和lateral connection。

- bottom-up 就是backbone 的不同stage
- top-down 就是深层次特征图上采样，使用的是最近邻上采样。这是最简单的一种插值方法，不需要计算，在待求像素的四个邻近像素中，将距离待求像素最近的邻近像素值赋给待求像素。
- lateral connection 有三个步骤
    - 对于每个stage输出的feature map，都先进行一个1*1的卷积降低维度。（降低到多少？）
    - 然后再将得到的特征和上一层采样得到特征图进行融合，就是直接相加，element-wise addition。因为每个stage输出的特征图之间是2倍的关系，所以上一层上采样得到的特征图的大小和本层的大小一样，就可以直接将对应元素相加 。
    - 相加完之后需要进行一个3*3的卷积才能得到本层的特征输出 。使用这个3*3卷积的目的是为了消除上采样产生的混叠效应(aliasing effect)，混叠效应应该就是指上边提到的‘插值生成的图像灰度不连续，在灰度变化的地方可能出现明显的锯齿状’
      在本文中，因为金字塔所有层的输出特征都共享classifiers/ regressors，所以输出的维度都被统一为256，即这些3*3的卷积的channel都为256。

![lateral_connection](https://github.com/RessCris2/cv_info/blob/main/imgs/fpn_2_lateral_connection.jpg)

## 为什么能解决问题？


## 代码参考


### FPN 出现点记录
- [实例分割新思路之SOLO v1&v2深度解析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/269598144)

- 原始的FPN会输出P2、P3、P4与P54个阶段的特征图，但在Mask RCNN中又增加了一个P6。将P5进行最大值池化即可得到P6，目的是获得更大感受野的特征，该阶段仅仅用在RPN网络中。


