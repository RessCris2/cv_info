

```python
# MMSegmentation 算法库目录结构的主要部分 
mmsegmentation 
   | 
   |- configs                        # 配置文件 
   |     |- _base_                   ## 基配置文件 
   |     |     |- datasets             ### 数据集相关配置文件 
   |     |     |- models               ### 模型相关配置文件 
   |     |     |- schedules            ### 训练日程如优化器，学习率等相关配置文件 
   |     |     |- default_runtime.py   ### 运行相关的默认的设置 
   |     |- swin                     ## 各个分割模型的配置文件，会引用 _base_ 的配置并做修改  
   |     |- ...                         
   |- data                           # 原始及转换后的数据集文件 
   |- mmseg  
   |     |- core                     ## 核心组件 
   |     |     |- evaluation           ### 评估模型性能代码 
   |     |- datasets                 ## 数据集相关代码 
   |     |     |- pipelines            ### 数据预处理 
   |     |     |- samplers             ### 数据集采样代码 
   |     |     |- ade.py               ### 各个数据集准备需要的代码 
   |     |     |- ... 
   |     |- models                    ## 分割模型具体实现代码 
   |     |     |- backbones             ### 主干网络 
   |     |     |- decode_heads          ### 解码头 
   |     |     |- losses                ### 损失函数 
   |     |     |- necks                 ### 颈 
   |     |     |- segmentors            ### 构建完整分割网络的代码 
   |     |     |- utils                 ### 构建模型时的辅助工具 
   |     |- apis                      ## high level 用户接口，在这里调用 ./mmseg/ 内各个组件 
   |     |     |- train.py              ### 训练接口 
   |     |     |- test.py               ### 测试接口 
   |     |     |- ... 
   |     |- ops                       ## cuda 算子（即将迁移到 mmcv 中） 
   |     |- utils                     ## 辅助工具 
   |- tools 
   |     |- model_converters          ## 各个主干网络预训练模型转 key 脚本 
   |     |- convert_datasets          ## 各个数据集准备转换脚本 
   |     |- train.py                  ## 训练脚本 
   |     |- test.py                   ## 测试脚本 
   |     |- ...                       
   |- ... 

```

segmentor, 一般包括 backbone、neck、head、loss 4 个核心组件，每个模块的功能如下：
1、预处理后的数据输入到 backbone 中进行编码并提取特征
2、输出的单尺度或者多尺度特征图输入到 neck 模块中进行特征融合或者增强，典型的neck是 FPN（特征金字塔，feature pyramid networks).
3、上述多尺度特征最终输入到 head 部分，一般包括 decoder head, auxiliary head 以及 cascade decoder head, 用以预测分割结果
4、最后一步是计算pixel分类的loss，进行训练。



fcn_head 是只完成上采样这一步吗？



## tensorboard
```python
vis_backends=[dict(type='LocalVisBackend'),
              dict(type='TensorboardVisBackend'),
              dict(type='WandbVisBackend')]
```


要求继承的多个文件中没有相同名称的字段，否则会报错。
常用的配置字段应该怎么配置？


## 日志
```python
# 在 default_runtime.py 中有定义
log_processor = dict(window_size=10, by_epoch=True, custom_cfg=None, num_digits=4)

```

写在文件里的继承，后面文件会修改前面出现过的key，或新增。
CUDA out of memory: 将batch size 改小。
