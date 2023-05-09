"""解析 voc 2012 的代码
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data


class VOC_Segmentation(Dataset):
 
    def __init__(self,root,text_name='train.txt',trans=None):
        super(VOC_Segmentation, self).__init__()
        #数据划分信息路径
        txt_path = os.path.join(root,'ImageSets','Segmentation',text_name)
        #图片路径
        image_path = os.path.join(root,'JPEGImages')
        #mask(label)路径
        mask_path = os.path.join(root,'SegmentationClass')
 
        #读入数据集文件名称
        with open(txt_path,'r') as f:
            file_names = [name.strip() for name in f.readlines() if len(name.strip()) > 0]
        #文件名拼接拼接路径
        self.images = [os.path.join(image_path,name+'.jpg') for name in file_names]
        self.mask = [os.path.join(mask_path,name+'.png') for name in file_names]
        self.trans = trans
 
    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, index):
        '''
        albumentations图像增强库是基于cv2库的,
        cv2.imread()读入后图片的类型是numpy类型
        所以需要保证与cv2读入类型一致
        '''
        img = np.asarray(Image.open(self.images[index]))
        mask = np.asarray(Image.open(self.mask[index]),dtype=np.int32)
 
        if self.trans is not None:
            img, mask = self.trans(img,mask)
 
        return img,mask



class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)


import albumentations as A
class Train_transforms():
    def __init__(self,output_size=480,scale_prob=0.5,flip_prob=0.5,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        # 来一波乱搞 transform?
        self.aug1 = A.Compose([
                #随机变换原图
                A.RandomScale(scale_limit=[0.5,1.5],interpolation=cv2.INTER_NEAREST,p=scale_prob),
                #小于480x480的img和mask进行填充
                A.PadIfNeeded(min_height=output_size,min_width=output_size,value=0,mask_value=255),
                #剪切
                A.RandomCrop(height=output_size,width=output_size,p=1),
                #翻转
                A.HorizontalFlip(p=flip_prob)
            ])
        self.aug2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])


    def __call__(self, image, mask):
        augmented = self.aug1(image=image, mask=mask)
        image, mask = augmented['image'],augmented['mask']



if __name__ == "__main__":
    voc_root = "/root/autodl-tmp/datasets"
    data = VOCSegmentation(voc_root)
    print('xxx')