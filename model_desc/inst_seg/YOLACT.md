# YOLACT
one-stage, anchor-based
real-time instance segmentation

## 提出背景
初步来看，是为了加速

## 具体方法
很惊艳，并且也获得了ICCV2019的COCO挑战赛创新奖。这篇文章是一个比较突破性想法的实例分割方法，是利用一组prototypes基底（也可以看做是一组字典），然后预测出一组线性组合系数，通过对这组prototypes基底进行线性组合，从而将实例凸显出来。这篇文章的一大优点就是：在进行instance的mask预测的时候，prototypes的线性组合系数与bounding box一一对应，从而在进行线性组合后，直接可以对组合后的mask进行crop和threshold。换句话说，就是一组线性组合系数就对应一个instance。

可以看出，虽然YOLACT原则上还并不能算作一个纯粹的单阶段实例分割器，但是这个想法已经可以将实例分割的速度大大地提升。虽然精度还没有Mask R-CNN高，但是有非常可观的发展前景。说到这里，就不得不提到YOLACT潜在存在的一个弊端，就是由于线性组合系数是一组数值，所以直接对离散的数值进行回归，深度神经网络做得并不好。而如果是做一个分类问题，深度神经网络就可以学得很好。这也是为什么Faster R-CNN系列都不是直接回归bounding box的x, y, w, h，而是将它们都编码为更好回归的编码后的形式。

作者：Brad Lucas
链接：https://www.zhihu.com/question/360594484/answer/934202464


breaking instance segmentation into two parallel subtasks: 
(1) generating a set of prototype masks
(2) predicting per-instance mask coefficients


同时提出 Fast NMS, a drop-in 12 ms faster replacement for standard NMS


