记录一些小工具的写法

## 判断 torch tensor 形状相同

## 
```python
def check_or_create(path):
    """
      If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)
```

## plot contours
https://scikit-image.org/docs/stable/auto_examples/edges/plot_contours.html#sphx-glr-auto-examples-edges-plot-contours-py
```python
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure


# Construct some test data
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp(np.sin(x)**3 + np.cos(y)**2))

# Find contours at a constant value of 0.8
contours = measure.find_contours(r, 0.8)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(r, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
```


```python
# 这个可以在轮廓处全是1， 其他，包括轮廓内部都是0.
def get_contours(img):
    """
        Returns only the contours of the image.
        The image has to be a binary image 
    """
    img[img > 0] = 1
    return dilation(img, disk(2)) - erosion(img, disk(2))
```


## 找到连通域，并且按照数字打标签

```python



```