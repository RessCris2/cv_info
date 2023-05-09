在 COCO 数据集上使用 mask rcnn 模型进行实例分割

将数据集转换为 coco 模式
pip install git+https://github.com/waspinator/pycococreator.git

参考这个代码，是否可以创建 coco 数据集？https://www.kaggle.com/code/junglecat2021/create-coco-dataset/notebook#Check-COCO
参考这个链接，跑通 demo https://zhuanlan.zhihu.com/p/344841580

# 使用官方数据集 COCO 跑数
需要的文件如下

- datasets/coco_instance.py
- 模型config文件：configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py
