本文档中主要记录常用到的目标检测和分割数据集以及信息描述。
主要围绕几个方面记录：
- 数据集url
- 注释
- 下载链接
- 标注格式

## 1. Cityscapes
**url**:数据描述和下载页面：https://www.cityscapes-dataset.com/dataset-overview/        
https://mp.weixin.qq.com/s? __biz=MzU2NjU3OTc5NA==&mid=2247560371&idx=1&sn=ab9a44e5d7e2af811c9ca209b12bbd09&chksm=fca9eb8ecbde6298b9072ce066459f00d15a939e92809c5b560048c4f73f1a4dc05041670d57&scene=27

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
Epithelial      

('Neoplastic'       
'Inflammatory'      
'Connective'    
'Dead'      
'Epithelial')       

MoNuSAC
Epithelial
Lymphocyte
Macrophage
Neutrophil


epithelial cells, lymphocytes, neutrophils,
and macrophages

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


## 8.MoNuSeg Dataset
Goal: Nuelci segmentation
Images: 29,000 nuclei from 44 images
Annotations: Pixel-level nuclei annotation
Download:https://monuseg.grand-challenge.org/Data/

## 9 DigestPath Dataset (MICCAI 2019)

Goal: Signet ring cell detection and colonoscopy tissue segmentation

Images: 77 images (2000px*2000px) from 20 WSIs for signet ring cell detection and 872 images (3000px*3000px) for colonoscopy tissue segmentation

Annotations: (1) bounding boxes for signet ring cells in xml format, (2) Pixel level annotation for colonoscopy tissue segmentation

Download: https://digestpath2019.grand-challenge.org/Home/ (access upon request)


## 10. ER+ Breast Cancer Dataset by Andrew Janowczyk
Goal: (1) ER+ breast cancer nuclei segmentation, (2) Epithelium region segmentation, (3) Lymphocyte detection, (4) Mitosis detection, (5) Invasive ductal carcinoma (IDC) identification, (6) Lymphoma subtype classification
Images: (1) 12,000 nuclei from 143 images (2000px*2000px) for nuclei segmentation, (2) 42 images (1000px*1000px) for epithelium segmentation, (3) 100 images (100px*100px) for lymphocyte detection, (4) 311 images (2000px*2000px) from 12 patients for mitosis detection, (5) 277,524 images (50px*50px) from 162 WSIs for IDC identification, (6) 374 images (1388px*1040px) for lymphoma subtype classification
Annotations: (1) Pixel-level nuclei annotation, (2) Pixel-level epithelium annotation, (3) The centers of 3,064 lymphocytes, (4) 550 mitosic centers, (5) binary label for IDC classification, (6) lymphoma subtype labels
Download (1): http://andrewjanowczyk.com/use-case-1-nuclei-segmentation/
Download (2): http://andrewjanowczyk.com/use-case-2-epithelium-segmentation/
Download (3): http://andrewjanowczyk.com/use-case-4-lymphocyte-detection/
Download (4): http://andrewjanowczyk.com/use-case-5-mitosis-detection/
Download (5): http://andrewjanowczyk.com/use-case-6-invasive-ductal-carcinoma-idc-segmentation/
Download (6): http://andrewjanowczyk.com/use-case-7-lymphoma-sub-type-classification/


## 11.GlAS Dataset (MICCAI 2015)
Goal: Gland segmentation
Images: 165 WSIs at x20 magnification
Annotations: Glandular boundaries
Download: https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/


## 12.MoNuSAC 
https://monusac-2020.grand-challenge.org/Data/

The dataset has over 46,000nuclei from37 hospitals,71patients, four organs, and four nucleus types

![MoNuSAC](https://github.com/RessCris2/cv_info/blob/main/imgs/monusac_dataset_1.png)

解析xml等代码 https://github.com/ruchikaverma-iitg/MoNuSAC




![mmseg_datasets](https://github.com/RessCris2/cv_info/blob/main/imgs/mmseg_datasets.png)