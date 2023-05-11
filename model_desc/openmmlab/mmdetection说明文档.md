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