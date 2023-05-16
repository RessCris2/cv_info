[documentation](https://mmsegmentation.readthedocs.io/en/latest/)
[教程2：准备数据集](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/user_guides/2_dataset_prepare.md)
[Docs > 基本概念 > 新增自定义数据集](https://mmsegmentation.readthedocs.io/zh_CN/latest/advanced_guides/add_datasets.html)

- 如果已经转成了 COCO dataset，是不是可以直接用？
- 官方还提供了一种较为简单的方式。如果还有更多的数据集可以采用这种。


可以参考官方 demo， mmsegmentation/demo/MMSegmentation_Tutorial.ipynb


## 将 pannuke 转换为 mmsegmentation 可以用的数据集格式，并配置好。

- 第一步是把数据集转换为 img, 保留原图，还有把mask也转换为 png 格式
- coco_format 文件夹下新建一个 seg_mask 用于保存转换后的 mask
```python
from tqdm import tqdm

def convert_pannuke(flag):
    masks = np.load("/root/autodl-tmp/datasets/pannuke/pn_format/{}/masks.npy".format(flag))
    output_dir = "/root/autodl-tmp/datasets/pannuke/coco_format/seg_mask/{}".format(flag)
    n = len(masks)
    for i in tqdm(range(n)):
        type_mask = np.argmax(masks[i, :, :, [5, 0, 1, 2, 3, 4]], axis=0)
        seg_filename = osp.join(output_dir, '{}.png'.format(i))
        Image.fromarray(type_mask.astype(np.uint8)).save(seg_filename, 'PNG')


flag = 'test'
convert_pannuke(flag)

flag = 'val'
convert_pannuke(flag)


```


## 第二步，指定类别什么的吧。
定义 Dataset.

```python
classname_to_id = {"Background": 0,
                   "Neoplastic": 1, 
                   "Inflammatory": 2,
                   "Connective/Soft tissue": 3, 
                   "Dead": 4,
                    "Epithelial": 5 
                   }


data_root = '/root/autodl-tmp/datasets/pannuke/coco_format'
img_dir = 'images'
ann_dir = 'seg_mask'
# define class and palette for better visualization
classes = ('Background', 'Background', 'Inflammatory', 'Connective/Soft tissue',
           'Dead', 'Epithelial')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123], [118, 20, 12]]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class PanNukeDataset(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
```


## 整合加训练
这里的话，fromfile 文件里面继承的数据集代码描述里也要写
dataset_type = 'PanNukeDataset'
data_root = '/root/autodl-tmp/datasets/pannuke/coco_format/'

```python
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine.runner import Runner
from mmengine import Config

@DATASETS.register_module()
class PanNukeDataset(BaseSegDataset):
    classes = ('Background', 'Inflammatory', 'Connective/Soft tissue',
              'Dead', 'Epithelial')
    palette = [[129, 127, 38], [120, 69, 125], [53, 125, 34], 
                [0, 11, 123], [118, 20, 12]]
    METAINFO = dict(classes = classes, palette = palette)
    def __init__(self, **kwargs):
      super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)

if __name__ == "__main__":
    cfg = Config.fromfile('/root/autodl-tmp/com_models/mmseg_demo/pannuke_unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py')
    print(f'Config:\n{cfg.pretty_text}')
    # Modify dataset type and path
    cfg.dataset_type = 'PanNukeDataset'
    cfg.data_root = '/root/autodl-tmp/datasets/pannuke/coco_format/'
    cfg.train_dataloader.batch_size = 2
    # Set up working dir to save files and logs.
    cfg.work_dir = '/root/autodl-tmp/com_models/mmseg_demo/pannuke_unet/work-dir'
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()
```