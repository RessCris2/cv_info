[轻松掌握 MMDetection 中 Head 流程](https://zhuanlan.zhihu.com/p/343433169)

需要注意的是：two-stage 或者 mutli-stage 算法，会额外包括一个区域提取器 roi extractor，用于将不同大小的 RoI 特征图统一成相同大小。    
虽然 head 部分的网络构建比较简单，但是由于正负样本属性定义、正负样本采样和 bbox 编解码模块都在 head 模块中进行组合调用，故 MMDetection 中最复杂的模块就是 head。


# Head 模块整体概述
# Head 模块构建流程
# Head 模块源码分析


每个 Head 内部都可能包括:
- RoI 特征提取器 roi_extractor
- 共享模块 shared_heads
- bbox 分类回归模块 bbox_heads
- mask 预测模块 mask_heads

其中1、3是必备模块。

```python
# Two stage 
#============= mmdet/models/detectors/two_stage.py/TwoStageDetector ============
def forward_train(...):
    # 先进行 backbone+neck 的特征提取
    x = self.extract_feat(img)
    losses = dict()
    # RPN forward and loss
    if self.with_rpn:
        # 训练 RPN
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                        self.test_cfg.rpn)
        # 主要是调用 rpn_head 内部的 forward_train 方法
        rpn_losses, proposal_list = self.rpn_head.forward_train(x,...)
        losses.update(rpn_losses)
    else:
        proposal_list = proposals
    # 第二阶段，主要是调用 roi_head 内部的 forward_train 方法
    roi_losses = self.roi_head.forward_train(x, ...)
    losses.update(roi_losses)
    return losses

# one stage
#============= mmdet/models/detectors/single_stage.py/SingleStageDetector ============
def forward_train(...):
    super(SingleStageDetector, self).forward_train(img, img_metas)
    # 先进行 backbone+neck 的特征提取
    x = self.extract_feat(img)
    # 主要是调用 bbox_head 内部的 forward_train 方法
    losses = self.bbox_head.forward_train(x, ...)
    return losses
```
two-stage Head 模块核心是调用 self.rpn_head.forward_train 和 self.roi_head.forward_train 函数，输出 losses 和其他相关数据。需要返回 proposal。
