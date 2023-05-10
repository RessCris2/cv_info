本文档中主要记录常用到的目标检测和分割数据集以及信息描述。
主要围绕几个方面记录：
- 数据集url
- 注释
- 下载链接
- 标注格式

## 1. Cityscapes
**url**:数据描述和下载页面：https://www.cityscapes-dataset.com/dataset-overview/
https://mp.weixin.qq.com/s?__biz=MzU2NjU3OTc5NA==&mid=2247560371&idx=1&sn=ab9a44e5d7e2af811c9ca209b12bbd09&chksm=fca9eb8ecbde6298b9072ce066459f00d15a939e92809c5b560048c4f73f1a4dc05041670d57&scene=27

**注释**：Cityscapes数据集，聚焦于城市街道场景的语义理解，是图像分割领域一个比较重要的数据集。这个数据有精细注释和粗糙的注释，还有视频。
可参考解释 https://blog.csdn.net/qq_34424944/article/details/129541001
Cityscape侧重于城市街道场景的语义理解。CityPersons数据集是cityscape的一个子集，是一个行人数据集，它只包含个人注释。有2975张图片用于培训，500张和1575张图片用于验证和测试。一幅图像中行人的平均数量为7人，提供了可视区域和全身标注。
也有一个 api 可以使用这个数据集：cityscapesscripts，https://github.com/mcordts/cityscapesScripts
gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip (11GB)。
- 5000张细粒度标注图像
- 20000张粗粒度标注图像

有19类数据？
在原始的gtFine数据集中就有的以labelIds结尾的数据：是所有类别的数据共有33类。
而在DeepLab论文中，只使用了其中19类，于是我们可以生成19类的数据集：以labelTrainIds结尾。

有一个博文中说图片大小为 2048*1024，待验证。

**下载链接**：CityPersons数据集下载：博主Rock_Huang~提供百度云链接地址：链接: 百度网盘  提取码：xyzj

**标注格式**：cityscapes中标注好的类别为VOC的标准格式（JPEGImages和Annotations）; 官方有提供一个脚本给图片添加label

## 2. PASCAL VOC

**url**:voc2012 链接：https://pan.baidu.com/s/1_KzASAbn3QLpcmBwgA0zaQ?pwd=xz12 提取码：xz12 

**注释**：
 -  20 object categories including vehicles, household, animals, and other: aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, TV/monitor, bird, cat, cow, dog, horse, sheep, and person.
- 1,464 images for training, 1,449 images for validation and a private testing set.

图片的像素尺寸大小不一，但是横向图的尺寸大约在500*375左右，纵向图的尺寸大约在375*500左右，基本不会偏差超过100。

**标注格式**：VOC的标准格式（JPEGImages和Annotations）



## 3. COCO-Stuff
**url**: https://github.com/nightrome/cocostuff

**注释**：164k images, pixel-level stuff annotations. These annotations can be used for scene understanding tasks like semantic segmentation, object detection and image captioning.

**下载链接**：http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip；

 http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

**标注格式**: 初步来看，stuff-only 和coco-style 一致，

## 4. COCO

**url**: https://cocodataset.org/#home

**注释**：Microsoft Common Objects in Context. 起源于微软于2014年出资标注的数据集。
包括91类目标 328000张影像和2500000（250万）个label.
目前为止有语义分割的最大数据集，提供的类别有80类，有超过33万张图片，其中20万张有标注，整个数据集中个体的数目超过150万个。

**标注格式**: 
object instances format
(针对别的任务  object keypoints, or image captions, format 分别是怎么样的？)
```
最终放进json文件里的字典
coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],   # 放一个空列表占位置，后面再append
    "annotations": []
}
annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 
```


# 医学图像
## 1. PanNuke
**url**: 
**注释**：
**下载链接**：
**标注格式**: 

## 2. kumar
**url**: 
**注释**：21623 nuclei, 30 images, 40x, 1000*1000
**下载链接**：
**标注格式**: 没有 class 标签，只是 instance map, Pixel-level nuclei annotation

## 3. CoNSep Dataset/consep
**url**: https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/

**注释**：Nuclei segmentation and classification; 24,319 exhaustively annotated nuclei with associated class labels; 41幅图，40x, 1000*1000


**下载链接**：https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/

**标注格式**: Pixel-level nuclei annotation with associated class labels
            - 这个数据集目前尬得慌，inst_id 是什么含义啊？





## 4. CPM17
**url**: https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
**注释**：Nuclei segmentation, 7570 nuclei; 32 幅图，40x and 20x, 500*500 to 600*600
**下载链接**：https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
**标注格式**: 没有 class 标签，只是 instance map, Pixel-level nuclei annotation

CoNSeP
Epithelial
Inflammatory
Spindle-Shaped
Miscellaneous

PanNuke
Neoplastic
Inflammatory
Connective
Dead
Non-Neoplastic Epithelial

MoNuSAC
Epithelial
Lymphocyte
Macrophage
Neutrophil


## 5. CRAG Dataset
MILD-Net: Colorectal Adenocarcinoma Gland (CRAG) Dataset
**url**: https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet/

**注释**: Gland instance segmentation; 213 images

**标注格式**:Gland instance-level ground truth

## 6.CRCHistoPhenotypes Dataset
Goal: Nuclei detection and classification

Images: 29,756 nuclei from 100 WSIs

Annotations: 29,756 nuclei centres out of which 22,444 with associated class labels

Download: https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe/


## 7.Lizard Colonic Nuclear Dataset
Goal: Colonic Nuclear Instance Segmentation and Classification