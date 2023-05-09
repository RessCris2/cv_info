MMDetection 和其他 OpenMMLab 仓库使用 MMEngine 的配置文件系统。 配置文件使用了模块化和继承设计，以便于进行各类实验。

```
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_{testing dataset information}.py
```

文件名分为五个部分。 每个部分用_连接，每个部分内的单词应该用-连接。

- {algorithm name}: 算法的名称。 它可以是检测器名称，例如 faster-rcnn、mask-rcnn 等。也可以是半监督或知识蒸馏算法，例如 soft-teacher、lad 等等

- {component names}: 算法中使用的组件名称，如 backbone、neck 等。例如 r50-caffe_fpn_gn-head 表示在算法中使用 caffe 版本的 ResNet50、FPN 和 使用了 Group Norm 的检测头。

- {training settings}: 训练设置的信息，例如 batch 大小、数据增强、损失、参数调度方式和训练最大轮次/迭代。 例如：4xb4-mixup-giou-coslr-100e 表示使用 8 个 gpu 每个 gpu 4 张图、mixup 数据增强、GIoU loss、余弦退火学习率，并训练 100 个 epoch。 缩写介绍:

- {gpu x batch_per_gpu}: GPU 数和每个 GPU 的样本数。bN 表示每个 GPU 上的 batch 大小为 N。例如 4x4b 是 4 个 GPU 每个 GPU 4 张图的缩写。如果没有注明，默认为 8 卡每卡 2 张图。

- {schedule}: 训练方案，选项是 1x、 2x、 20e 等。1x 和 2x 分别代表 12 epoch 和 24 epoch，20e 在级联模型中使用，表示 20 epoch。对于 1x/2x，初始学习率在第 8/16 和第 11/22 epoch 衰减 10 倍；对于 20e ，初始学习率在第 16 和第 19 epoch 衰减 10 倍。

- {training dataset information}: 训练数据集，例如 coco, coco-panoptic, cityscapes, voc-0712, wider-face。

- {testing dataset information} (可选): 测试数据集，用于训练和测试在不同数据集上的模型配置。 如果没有注明，则表示训练和测试的数据集类型相同。