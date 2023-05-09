## APNB（Asymmetric Pyramid Non-local Block）

leverages a pyramid sampling module into the non-local block

大幅降低运算和内存负荷，而不降低效果。

## Asymmetric Fusion Non-local Block (AFNB)

fuse the features of different levels under a sufficient consideration of long range dependencies

**效果**： 81.3 mIoU

数据集： Cityscapes test set

256x128 input, APNB is around 6 times faster than a non-local block on GPU while 28 times smaller in GPU running memory occupation

![ann_1](https://github.com/RessCris2/cv_info/blob/main/imgs/ann_1.png)


问题：

non-local block 提出的背景是什么？为什么能够解决内存和运算负荷，却保持效果？
从这里看， AFNB， APNB 两个部分都用到了 attention 的概念。