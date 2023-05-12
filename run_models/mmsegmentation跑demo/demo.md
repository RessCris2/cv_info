使用 mmsegmentation 跑一个语义分割的案例。

# 使用 PASCAL VOC2012 跑模型
mmsegmentation 中现有的可用checkpoint: https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/model_zoo.md 中点开相应模型链接可以找到。   
模型训练命令    
```
python tools/train.py  ${CONFIG_FILE} [optional arguments]
```

假如在 PASCAL VOC2012 上训练一个 deeplabv3+.    
需要找到 config 文件, 这里 config 的命名规则为：    
```
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_{testing dataset information}
```
参考链接中的说明：https://mmsegmentation.readthedocs.io/en/latest/user_guides/1_config.html#config-name-style

选择config文件：
- /root/autodl-tmp/com_models/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-20k_voc12aug-512x512.py    
- schedule config file: /root/autodl-tmp/com_models/mmsegmentation/configs/_base_/schedules/schedule_20k.py



# 使用cityscapes数据集在mmsegmentation框架下训练语义分割模型    

参考链接 https://blog.csdn.net/m0_46201134/article/details/128156430        
关键在于 cityscapes 中的数据准备部分。https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py    

安装 python -m pip install cityscapesscripts    