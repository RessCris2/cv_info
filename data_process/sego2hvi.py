# -*- coding:utf-8 -*-
""" 将segmentation models 的预测结果转换为 hvi 的格式
    【N, C, H, W]  --> [img, inst, type]
"""
import torch
import numpy as np
import os
import sys
import glob
import cv2
from scipy import ndimage
import pathlib
from tqdm import tqdm
# sys.path.append("/root/autodl-tmp")
# from seg_models.play_sem import  PanNukeDataset


def mask2polygon(mask):
    # 这里我改为只返回外部轮廓，且轮廓点数大于4
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour) > 4:# and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    
    # if len(segmentation) == 0:
    #     raise "some error!"
    return segmentation


def infer_image(model_path, images_dir, save_dir, threshold=0.3 ):
    """ 先一张一张图片预测吧
        如果单纯取 max 的话就不需要 threshold 了。。
        threshold 之后取max?
    """
    # model_path = "/root/autodl-tmp/seg_models/best_model.pth"
    model = torch.load(model_path)

    img_path = os.path.join(images_dir, "*.jpg")
    img_list = glob.glob(img_path)
    assert len(img_list) > 0, "0 imgs to infer!"
    for img in tqdm(img_list):
        base_name = pathlib.Path(img).stem 
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_pred = np.transpose(image, (2, 0, 1))[None, ...]
        pred = model(torch.tensor(image_pred, dtype=torch.float32).to("cuda")).cpu().detach().numpy().squeeze(0)

        # 将没有超过一定分数的位置全都置 0
        pred = np.where(pred >= 0.3, pred, 0)  #  [H, W], 是类别mask

        # 取argmax
        type_mask = np.argmax(pred ,axis=0)

        
        inst_mask = type2inst_mask(type_mask)
        final = np.c_[image, inst_mask[...,None], type_mask[...,None]]

        save_path = os.path.join(save_dir, base_name)
        np.save(save_path, final)

    # return pred_mask

def type2inst_mask(type_mask):
    """在类别 mask 的基础上, 获取 inst_mask
      即找到n个 instance, 赋不同的 instance_id
    """ 
    cnt = 0
    inst_mask = np.zeros([256, 256])

    # 剔除第一维 背景的预测
    for c in range(1, type_mask.shape[0]):
        contours, hierarchy = cv2.findContours((np.where(type_mask == c, type_mask, 0)).astype(np.uint8), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #     print(c, len(contours))
        for i, contour in enumerate(contours):
            if len(contour) > 4:
                cnt += 1
                inst_mask = cv2.fillPoly(inst_mask, [contour], cnt)
    return inst_mask


def transfer2hvi(model_path, images_dir, num_classes=6):
    """
      model_path: 加载模型
      infer_dir: 对图像进行预测， 图像的格式是 jpg
    """
    model = torch.load(model_path)
    
    img_path = os.path.join(images_dir, "*.jpg")
    img_list = glob.glob(img_path)
    for img in img_list:
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        pred = model(torch.tensor(image[None, ...], dtype=torch.float32))
        # pred = model(image)  # (1, 6, H, W)  -- > (img, inst, type)

      
        pred = torch.argmax(pred.squeeze(), axis=0)  ## 找到最大的维度作为该像素的预测么？如果都很小咋办？
        # masks = [(mask == v) for v in self.class_values]
        pred = np.stack(pred, axis=-1).astype('float')
        preds = np.transpose(pred, (2, 0, 1)) 


if __name__ == "__main__":
    #
    # model_path = "/root/autodl-tmp/seg_models/best_model_v2.pth"
    # model = torch.load(model_path)

    # img_path = '/root/autodl-tmp/datasets/pannuke/coco_format/images/train/0.jpg'
    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = np.transpose(image, (2, 0, 1))[None, ...]
    # pred = model.predict(torch.tensor(image, dtype=torch.float32).to("cuda")).cpu().numpy().squeeze(0)
    # print(pred)
    # # pred = model(torch.tensor(image, dtype=torch.float32)).cpu().numpy().squeeze(0)

    # # 取argmax
    # pred_mask = np.argmax(pred ,axis=0)

    # # 将没有超过一定分数的位置全都置 0
    # pred_mask = np.where(pred >= 0.3, pred_mask, 0)  #  [H, W], 是类别mask
    

    model_path = "/root/autodl-tmp/seg_models/best_model.pth"
    images_dir = '/root/autodl-tmp/datasets/pannuke/coco_format/images/test'
    save_dir = '/root/autodl-tmp/datasets/pn_unet_infer/hvi_format'
    infer_image(model_path, images_dir, save_dir, threshold=0.3)
    

    

    



