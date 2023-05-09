# Segmentaion标签的三种表示：poly、mask、rle
参考链接 https://blog.csdn.net/weixin_44966641/article/details/123171026

[Create your own COCO-style datasets](https://patrickwasp.com/create-your-own-coco-style-dataset/)

不同于图像分类这样比较简单直接的计算机视觉任务，图像分割任务（又分为语义分割、实例分割、全景分割）的标签形式稍为复杂。在分割任务中，我们需要在像素级上表达的是一张图的哪些区域是哪个类别。

## 多边形坐标Polygon
[polygon 表达](https://github.com/RessCris2/cv_info/blob/main/imgs/poly.jpg)

可以使用 cv2.polylines 画图
```python
import numpy as np
import cv2
cat_poly = [[390.56410256410254, 1134.179487179487], 
            # ...
            [407.2307692307692, 1158.5384615384614]]

dog_poly = [[794.4102564102564, 635.4615384615385], 
            # ...
            [780.3076923076923, 531.6153846153846]]

img = cv2.imread("cat-dog.jpeg")

cat_points = np.array(cat_poly, dtype=np.int32)
cv2.polylines(img, [cat_points], True, (255, 0, 0), 3)
dog_points  = np.array(dog_poly, dtype=np.int32)
cv2.polylines(img, [dog_points], True, (0, 0, 255), 3)

cv2.imshow("window", img)
cv2.waitKey(0)


```

## 掩码 mask
![mask 表达](https://github.com/RessCris2/cv_info/blob/main/imgs/poly.jpg)

## rle

```
mask=np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )

```
mask 表达是有很多信息冗余的，因为只有0，1两种元素，RLE编码就是将相同的数据进行压缩计数，同时记录当前数据出现的初始为位置和对应的长度，例如：[0,1,1,1,0,1,1,0,1,0] 编码之后为1,3,5,2,8,1。其中的奇数位表示数字1出现的对应的index，而偶数位表示它对应的前面的坐标位开始数字1重复的个数。
RLE全称（run-length encoding），翻译为游程编码，又译行程长度编码，又称变动长度编码法（run coding），在控制论中对于二值图像而言是一种编码方法，对连续的黑、白像素数(游程)以不同的码字进行编码。游程编码是一种简单的非破坏性资料压缩法，其好处是加压缩和解压缩都非常快。其方法是计算连续出现的资料长度压缩之。

RLE是COCO数据集的规范格式之一，也是许多图像分割比赛指定提交结果的格式。

# 编码之间的变换

## poly 转 mask

```python
def poly2mask(points, width, height):
    mask = np.zeros((width, height), dtype=np.int32)
    obj = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, obj, 1)
    return mask
```

## mask 转 rle
使用 pycocotools 工具包
```python

## method1, pycocotools
def singleMask2rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

## method2, pycococreatortools
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle
```
该函数的返回值 rle 是一个字典，有两个字段 size 和 counts ，该字典通常直接作为 COCO 数据集的 segmentation 字段.


## mask 转 poly
会有精度损失

```python


## method1 
from skimage import measure

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

def get_paired_coord(coord):
    points = None
    for i in range(0, len(coord), 2):
        point = np.array(coord[i: i+2], dtype=np.int32).reshape(1, 2)
        if (points is None): points = point
        else: points = np.concatenate([points, point], axis=0)
    return points

## method2: pycococreatortools
def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons
```


## rle 转 mask
```python
def rle_decode(mask_rle, shape=(520, 704)):
    s = mask_rle.split()
    starts =  np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)

    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def rle_decode(mask_rle, shape=(520, 704), color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    # Split the string by space, then convert it into a integer array
    s = np.array(mask_rle.split(), dtype=int)

    # Every even value is the start, every odd value is the "run" length
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    # The image image is actually flattened since RLE is a 1D "run"
    if len(shape)==3:
        h, w, d = shape
        img = np.zeros((h * w, d), dtype=np.float32)
    else:
        h, w = shape
        img = np.zeros((h * w,), dtype=np.float32)

    # The color here is actually just any integer you want!
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
        
    # Don't forget to change the image back to the original shape
    return img.reshape(shape)
```

## 其他工具代码
```python

## 找到相应的 bounding box
from pycocotools import mask
bounding_box = mask.toBbox(binary_mask_encoded)
```