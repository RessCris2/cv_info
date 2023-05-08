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

**下载链接**：CityPersons数据集下载：博主Rock_Huang~提供百度云链接地址：链接: 百度网盘  提取码：xyzj
**标注格式**：cityscapes中标注好的类别为VOC的标准格式（JPEGImages和Annotations）

## 2. PASCAL VOC
**url**:voc2012 链接：https://pan.baidu.com/s/1_KzASAbn3QLpcmBwgA0zaQ?pwd=xz12 提取码：xz12 
**注释**：
**标注格式**：VOC的标准格式（JPEGImages和Annotations）

## 3. COCO-Stuff
**url**: https://github.com/nightrome/cocostuff
**注释**：164k images, pixel-level stuff annotations. These annotations can be used for scene understanding tasks like semantic segmentation, object detection and image captioning.
**下载链接**：http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip；
 http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
**标注格式**: 初步来看，stuff-only 和coco-style 一致，


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
**标注格式**: 

## 3. CoNSep Dataset/consep
**url**: https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
**注释**：Nuclei segmentation and classification; 24,319 exhaustively annotated nuclei with associated class labels; 41幅图，40x, 1000*1000
**下载链接**：https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
**标注格式**: Pixel-level nuclei annotation with associated class labels


## 4. CPM17
**url**: https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
**注释**：Nuclei segmentation, 7570 nuclei; 32 幅图，40x and 20x, 500*500 to 600*600
**下载链接**：https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
**标注格式**: Pixel-level nuclei annotation with associated class labels

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