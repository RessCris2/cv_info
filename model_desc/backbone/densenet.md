

esNet模型的核心是通过建立前面层与后面层之间的“短路连接”（shortcuts，skip connection），这有助于训练过程中梯度的反向传播，从而能训练出更深的CNN网络.
基本思路与ResNet一致，但是它建立的是前面所有层与后面层的密集连接（dense connection）.DenseNet的另一大特色是通过特征在channel上的连接来实现特征重用（feature reuse）。这些特点让DenseNet在参数和计算成本更少的情形下实现比ResNet更优的性能.


[DenseNet：比ResNet更优的CNN模型](https://zhuanlan.zhihu.com/p/37189203)
评论区说显存占用太大，不实用。
