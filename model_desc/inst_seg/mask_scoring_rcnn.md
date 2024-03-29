作者： 华科和地平线
年份： 2019
会议： CVPR

[[CVPR2019]:Mask Scoring R-CNN](https://zhuanlan.zhihu.com/p/58291808)

## 提出背景

从实例分割中mask 的分割质量角度出发，提出过去的经典分割框架存在的一个缺陷：用Bbox bounding box的classification confidence作为mask score，导致mask score和mask quality不配准。
基于Mask R-CNN提出一个新的框架Mask Scoring R-CNN，能自动学习出mask quality，试图解决不配准的问题。    
Mask R-CNN存在的问题是：bounding box的classification confidence不能代表mask的分割质量。classification confidence高可以表示检测框的置信度高（严格来讲不能表示框的定位精准），但也会存在mask分割的质量差的情况。高的分类置信度也应该同时有好的mask 结果。   
初衷： 望得到精准的mask质量，那么如何评价输出的mask质量呢？
> 是AP，或者说是instance-level的IoU。这个IoU和检测用到的IoU是一个东西，前者是predict mask和gt mask的pixel-level的Intersection-over-Union，而后者则是predict box和gt box的box-level的Intersection-over-Union。   
所以一个直观的方法就是用IoU来表示分割的质量，那么让网络自己学习输出分割的质量也是简单直观的做法。   
学习出mask的IoU，那么最后的mask score就等于maskIoU乘以classification score，mask score就同时表示分类置信度和分割的质量。


这段话的意思是 学习 mask iou值？gt为真实的 mask iou?
## 具体方法