# -*- coding:utf-8 -*-
"""
    将 mmdetection 中的预测结果转换成 HoVerNet 中适用的评估输入方式。
    有一种方式就是先转换成 pannuke 的格式，然后再用之前的方式转换

    所以第一步先把 infer 的结果返回为 [256, 256, 5]
    我现在要找 infer 那一步
"""
from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import torch
import pathlib
import os
import glob
from tqdm import tqdm
import cv2

# 指定模型的配置文件和 checkpoint 文件路径
# config_file = '/root/autodl-tmp/work_dirs/mask_rcnn_r50_fpn_1x_pannuke/mask_rcnn_r50_fpn_1x_pannuke.py'
# checkpoint_file = '/root/autodl-tmp/work_dirs/mask_rcnn_r50_fpn_1x_pannuke/latest.pth'
# img = '/root/autodl-tmp/mmdet_project/dataset/pannuke/training_data/images/train/0.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次


def maskrcnn_result(result, score_thr=0.3):
        """
        Args:
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.

        Returns:
        """
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
       
        # draw bounding boxes
        if score_thr > 0:
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]

        return bboxes, labels, segms
      
def transfer2npy(labels, segms):
    """
        最后输出的 npy, 保存到相应路径中

    """
    # channel 几个类的mask，得到 [256, 256, 5]
    # n_mask = []
    # t_mask = []
    type_mask = np.zeros(shape=(256, 256))
    inst_mask =  np.zeros(shape=(256, 256))

    cnt = 0
    for c_id in range(1, 5): # class_name 对应的id
        c_mask = []
        if segms is None:
            print('xxxx')
        for i in range(len(segms)):
            if labels[i] == c_id:
                cnt += 1
                seg = segms[i].astype(np.uint8)
                type_mask = np.where(seg > 0, labels[i], type_mask)
                inst_mask = np.where(seg > 0, cnt, inst_mask)
                # c_mask.append(seg)
            
        
        
        # if len(c_mask) > 0:
        #     mask = np.array(c_mask).sum(axis=0)
            
        #     # plt.imshow(mask)
        #     # print(c_id)
        # else:
        #     # 假如某一类没有对应的 seg， 则生成一个都是0的mask
        #     mask = np.zeros(shape=(256, 256))
        
        # n_mask.append(mask)
        
    # final_mask = np.c_[n_mask]
    # inst_mask = n_mask.sum(axis=-1)
    # final_mask = np.transpose(final_mask, (1, 2, 0))

    # 
    # type_mask = t_mask.sum(axis=-1)
    return inst_mask, type_mask
        
def infer_one_img(model, img_path, output_dir):
    
    # img = "/root/autodl-tmp/mmdet_project/play_mmdetection/000000000785.jpg"
    img = cv2.imread(img_path)[:, :, ::-1] # 转换文 rgb
    result = inference_detector(model, img_path)
    bboxes, labels, segms = maskrcnn_result(result)
    if len(labels) == 0:
        # 如果没有任何输出，则都是0
        inst_mask = np.zeros(shape=(256, 256))
        type_mask = np.zeros(shape=(256, 256))
    else:
        inst_mask, type_mask = transfer2npy(labels, segms)
    
    

    final_mask = np.c_[img, inst_mask[:,:,None], type_mask[:,:,None]]
    base_name = pathlib.Path(img_path).stem 
    save_path = os.path.join(output_dir, "{}.npy".format(base_name))
    np.save(save_path, final_mask)

def infer_dir(model, input_dir, output_dir):


    file_path = os.path.join(input_dir, "*.jpg")
    file_list = glob.glob(file_path)
    
    # file_list = glob.glob(input_dir + "/*.jpg")
    file_list.sort()  # ensure same order [1]
    assert len(file_list) >0, "no files to infer!"
    for file in tqdm(file_list):
        infer_one_img(model, file, output_dir)

   

if __name__ == "__main__":
    config_file = '/root/autodl-tmp/work_dirs/mask_rcnn_r50_fpn_1x_pannuke/mask_rcnn_r50_fpn_1x_pannuke.py'
    checkpoint_file = '/root/autodl-tmp/work_dirs/mask_rcnn_r50_fpn_1x_pannuke/latest.pth'
    # 根据配置文件和 checkpoint 文件构建模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    input_dir = '/root/autodl-tmp/datasets/pannuke/coco_format/images/test/'
    output_dir = '/root/autodl-tmp/datasets/pn_maskrcnn_infer/20230324/hvi_format'
    infer_dir(model, input_dir, output_dir)