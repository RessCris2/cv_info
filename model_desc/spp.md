# spatial paramid pooling

空间金字塔汇集法
参考资料
- 解读SPPNet https://zhuanlan.zhihu.com/p/545339758



Atrous Spatial Pyramid Pooling

## 研究背景（这篇论文主要解决什么问题）

SPPNet发表于RCNN之后，Fast RCNN之前. 针对RCNN的计算重复问题（SS算法的k个proposal需要经过k次CNN的计算），SPPNet提出了Spatial pyramid pooling（SPP），使得k个proposal只需经过1次CNN的计算，就可以获得固定尺寸的输出，这极大提高的计算效率.

解决了CNN需要固定image输入尺寸的问题. 该方法使得CNN的初始输入image不再需要缩放至固定尺寸，也可以在该层之后获得固定尺寸的输出. 这使得模型使用多尺度输入进行训练成为可能，而多尺度训练可以有效降低模型过拟合. 同时，该方法基于不同kernel_size的pooling，还能够获得多尺度的特征. 此外，基于SPP，可以避免了RCNN中proposal的重复计算问题. 整张image仅需一次CNN的计算，极大提高了计算效率和速度.

## 具体方法

当SPP的输入特征图通道数为256时，对其进行不同kernel_size和stride的max pooling操作，得到4x4x256、2x2x256和1x1x256大小的输出. 将上述三个输出view(-1)成4096(16x256)、1024(4x256)和256(1x256)大小，并按照进行拼接，即可以得到长度为5376的vector. 由于输出长度固定，所以可以作为后续FC层的输入. 那么，重点是如何根据不同size的feature map计算所需max pooling的kernel_size和stride？假设输入特征图的为axa大小，则计算方式是
$$ k = celing(\frac{a}{n}), s = floor(\frac{a}{n})$$
，其中
n 为输出的特征图尺寸，ceiling指向上取整，floor指向下取整.
SPPNet会将image输入resize成方形. 边长为image宽高的最小值. 即S = min(H, W).
## 为什么能解决这个问题

## 简单代码
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPP(nn.Module):
    def __init__(self, output_size: list):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, x):
        b, _, h, w = x.shape
        outputs = [F.max_pool2d(x, kernel_size=(math.ceil(h/self.output_size[i]), math.ceil(w/self.output_size[i])), 
                                stride=(math.floor(h/self.output_size[i]), math.floor(w/self.output_size[i]))).view(b, -1)
                   for i in range(len(self.output_size))]
        print([f'branch{i} output shape: {x.shape}' for i, x in enumerate(outputs)])
        return torch.cat(outputs, dim=1)
   
   
inputs = torch.randn(4, 256, 10, 20)
print('--- classification config ---')
spp = SPP(output_size=[4, 2, 1])
outputs = spp(inputs)
print(outputs.shape, '\n')
print('--- object detection config ---')
spp = SPP(output_size=[6, 3, 2, 1])
outputs = spp(inputs)
print(outputs.shape)
```
