本文档想对 mmdetection 的学习做些记录

[mmdetection话题](https://www.zhihu.com/topic/21544084/hot) 里面的内容相对比较散    
[不得不知的 MMDetection 学习路线(个人经验版)](https://zhuanlan.zhihu.com/p/369826931) 官方在知乎发布的一些指导，里面有一系列链接，可以按需使用。    
[轻松掌握 MMDetection 整体构建流程(一)](https://zhuanlan.zhihu.com/p/337375549) 可以作为 目标检测以及mmdetection 的总览（可以面试备用hh)    


![目标检测算法分类](https://github.com/RessCris2/cv_info/blob/main/imgs/mmdet_classify_1.jpg)
## 目标检测算法核心组件划分
![MMDetection 代码构建流程](https://github.com/RessCris2/cv_info/blob/main/imgs/mmdet_train_procedure_1.jpg)
参照这份流程，对每个算法进行剖析。


训练部分一般包括 9 个核心组件，总体流程是：

- 任何一个 batch 的图片先输入到 backbone 中进行特征提取，典型的骨干网络是 ResNet
- 输出的单尺度或者多尺度特征图输入到 neck 模块中进行特征融合或者增强，典型的 neck 是 FPN
- 上述多尺度特征最终输入到 head 部分，一般都会包括分类和回归分支输出
- 在整个网络构建阶段都可以引入一些即插即用增强算子来增加提取提取能力，典型的例如 SPP、DCN 等等
- 目标检测 head 输出一般是特征图，对于分类任务存在严重的正负样本不平衡，可以通过正负样本属性分配和采样控制
- 为了方便收敛和平衡多分支，一般都会对 gt bbox 进行编码
- 最后一步是计算分类和回归 loss，进行训练
- 在训练过程中也包括非常多的 trick，例如优化器选择等，参数调节也非常关键

在网络构建方面，理解目标检测算法主要是要理解 head 模块。    
MMDetection 中 head 模块又划分为 two-stage 所需的 RoIHead 和 one-stage 所需的 DenseHead，也就是说所有的 one-stage 算法的 head 模块都在mmdet/models/dense_heads中，而 two-stage 算法还包括额外的mmdet/models/roi_heads。   

## 目标检测核心组件功能

mmcv： MMCV 2.0 归纳总结了各个算法数据变换方向的需求，实现了一系列功能强大的数据变换。
mmengine

数据加载、评测、

发自己的项目，会发现要想开启混合精度训练，需要同时配置多个模块，例如给模型设置 fp16_enabled 、启用 Fp16OptimizerHook，还需要给模型的各个接口加上类似 auto_fp16 的装饰器，少写一处都会无法顺利开启混合精度训练。

训练引擎、评估引擎和模块管理

执行器是 MMEngine 中所有模块的“管理者”。所有的独立模块——不论是模型、数据集这些看得见摸的着的，还是日志记录、分布式训练、随机种子等相对隐晦的——都在执行器中被统一调度、产生关联。事物之间的关系是复杂的，但执行器为你处理了一切，并提供了一个清晰易懂的配置式接口。



## 使用 Tips
### Tensorboard log
```python

# default_runtime.py
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)
```

### 如何调整学习率？
schedule_xxx.py 里面设置了 lr_config


```python
# 默认的设置如下

# optimizer
optimizer = dict(type='SGD', lr=0.02/8, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
```

### 如何修改 coco 的类别等数据（mmdet3.0)
2.x 试过只有改原始文件 coco.py 才生效。

```python


```



## 返回结果 inference
bbox and mask

```python
import numpy as np
from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
import pathlib
import os
import glob
from tqdm import tqdm
import cv2
def maskrcnn_result(result, score_thr=0.3):
        """
        Args:
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.

        Returns:
        """
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
       
        # draw bounding boxes
        if score_thr > 0:
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]

        return bboxes, labels, segms
```


## 版本问题

mmdetection3.x
之前安装的是2.x，发现 mmengine 版本变化对不上。重装试试。

留意和hook相关的细节


```python
# 1
@HOOKS.register_module()
class CheckpointHook(Hook)

# 2
@VISBACKENDS.register_module()
class TensorboardVisBackend(BaseVisBackend)

# 3
@EVALUATOR.register_module()
class Evaluator:





```



## 效果评估

pip install git+https://github.com/cocodataset/panopticapi.git
 
[COCO API 的评估方式](https://cocodataset.org/#detection-eval)