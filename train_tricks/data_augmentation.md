
# 问题

## ToTensor()

> transforms.ToTensor()函数的作用是将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型。
输入模式为（L、LA、P、I、F、RGB、YCbCr、RGBA、CMYK、1）的PIL Image 或 numpy.ndarray (形状为H x W x C)数据范围是[0, 255] 到一个 Torch.FloatTensor，其形状 (C x H x W) 在 [0.0, 1.0] 范围内。


## from skimage import img_as_ubyte

可以将 float 转换为[0, 255]
```python
def prepare_prob(img, convertuint8=True, inverse=True):
    """
    Prepares the prob image for post-processing, it can convert from
    float -> to uint8 and it can inverse it if needed.
    """
    if convertuint8:
        img = img_as_ubyte(img)
    if inverse:
        img = 255 - img
    return img
```

