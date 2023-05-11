作者： 华科和地平线
年份： 2019
会议： CVPR

[[CVPR2019]:Mask Scoring R-CNN](https://zhuanlan.zhihu.com/p/58291808)

## 提出背景

从实例分割中mask 的分割质量角度出发，提出过去的经典分割框架存在的一个缺陷：用Bbox bounding box的classification confidence作为mask score，导致mask score和mask quality不配准。
基于Mask R-CNN提出一个新的框架Mask Scoring R-CNN，能自动学习出mask quality，试图解决不配准的问题。    
Mask R-CNN存在的问题是：bounding box的classification confidence不能代表mask的分割质量。classification confidence高可以表示检测框的置信度高（严格来讲不能表示框的定位精准），但也会存在mask分割的质量差的情况。高的分类置信度也应该同时有好的mask 结果。   
初衷： 望得到精准的mask质量，那么如何评价输出的mask质量呢？

## 具体方法