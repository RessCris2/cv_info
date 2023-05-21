
# 参考
- [bulling 代码复现](https://github.com/bubbliiiing/unet-pytorch)



unet 的backbone 使用 vgg16, resnet50 作为下采样路径，都有5个阶段，每个阶段的输出形状并不一样吧。(待确认)
224 --> 112 --> 56 --> 28 --> 14
224 --> 56 -- > 28 --> 14 --> 7 

UNet 对输入形状有要求吗？

输入图片的大小, 32的倍数