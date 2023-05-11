# SOLO

参考 
- [实例分割新思路之SOLO v1&v2深度解析](https://zhuanlan.zhihu.com/p/269598144)

## 提出背景


## 具体方法
 instance segmentation is decomposed into two classification tasks

 一阶段的实例分割方法
 这篇文章很巧妙的地方就在于Mask Branch的每一张特征图有且唯一对应一个Category Branch的位置（i, j），并且在Mask Branch的S^2通道中，每一个通道有且仅有一个instance。这样，在训练的时候，就可以通过loss将Category和Mask两个分支关联起来。SOLO这个名称很有向YOLO致敬的味道，因为YOLO当时突破Faster R-CNN系列网络时，做法也是将一张图分割成m*n个格子，然后一次性地对物体进行预测（而不需要先用RPN来对建议区域进行提取）。所以SOLO这种单阶段实例分割网络对以往两阶段实例分割网络的突破就在于，他也引入了这种You only look once的机制。这样的话，我觉得YOLO有的一些缺点可能也会应验到SOLO身上，比如，当两个instance的中心都投影到同一个格子中时，SOLO只能识别出一个instance，从而忽略掉另一个。

作者：Brad Lucas
链接：https://www.zhihu.com/question/360594484/answer/934202464
来源：知乎
