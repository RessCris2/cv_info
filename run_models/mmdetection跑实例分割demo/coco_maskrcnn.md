在 COCO 数据集上使用 mask rcnn 模型进行实例分割

将数据集转换为 coco 模式
pip install git+https://github.com/waspinator/pycococreator.git

参考这个代码，是否可以创建 coco 数据集？https://www.kaggle.com/code/junglecat2021/create-coco-dataset/notebook#Check-COCO
参考这个链接，跑通 demo https://zhuanlan.zhihu.com/p/344841580

# 使用官方数据集 COCO 跑数
需要的文件如下

- datasets/coco_instance.py
- 模型config文件：configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py








## 如何解读跑数结果？(这里用的是 consep_cascade_mask 的例子)
做推断的脚本

```python
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = '/root/autodl-tmp/com_models/mmdet_demo/consep_cascade/cascade_mask_rcnn_r50_fpn_1x_coco_consep.py'
checkpoint_file = '/root/autodl-tmp/com_models/mmdet_demo/consep_cascade/work_dir/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a single image
img = '/root/autodl-tmp/datasets/consep/images/test/000.jpg'
result = inference_detector(model, img)
# show the results
show_result_pyplot(model, img, result)
```

result