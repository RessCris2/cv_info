# -*- coding:utf-8 -*-
import numpy as np
import os
import sys
sys.path.append("/root/autodl-tmp/")
import pandas as pd
import cv2
# from tools import mask2polygon, NpEncoder
import json
from tqdm import tqdm

from utils.tool import mk_file

def find_bbox(ma):
        """
            如果输入只有一个instance, 就应该只返回一个 bbox
        """
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(ma.astype(np.uint8))
        stats = stats[stats[:, 4].argsort()]
        return stats[:-1, :4] # 去除背景

def get_bbox(mask):
    ## 根据 mask 的取值确定 instance 数量； 也就是获取一个mask 上的多个instance
    ##
    # 对每个单独的instance获取bbox
    vals = np.unique(mask)
    bboxes = []
    for val in vals:
        if val != 0:  # 去除0，背景
            ma = np.where(mask == val, 1, 0)
            bbox = find_bbox(ma)
            bboxes.extend(bbox)
    return bboxes

def box_xywh2_xyxy(boxes):
    """
    boxes: 双重列表？
    Args:
        boxes:

    Returns:

    """

    new_bboxes = []
    for box in boxes:
        xmin, ymin, w, h = box
        box1 = [[xmin, ymin, xmin+w, ymin+h]]
        new_bboxes.extend(box1)
    return new_bboxes

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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def transfer2coco(img_path_base, masks_path, json_file ):
    masks = np.load(masks_path)
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    bnd_id = 1
    image_id= 0
    
    # category id 要从0开始吗？
    # classname_to_id = {"background": 0, "class_0": 1, "class_1": 2, "class_2": 3, "class_3": 4, "class_4": 5}
    classname_to_id = {"Background": 0, "Neoplastic": 1, "Inflammatory": 2, "Connective/Soft tissue": 3, "Dead": 4, "Epithelial": 5 }

    # 原数据中的channel顺序为 {"Neoplastic": 0, "Inflammatory": 1, "Connective/Soft tissue": 2, "Dead": 3, "Epithelial": 4, "Background": 5}
    # 要转换为  {"Background": 0，"Neoplastic": 1, "Inflammatory": 2, "Connective/Soft tissue": 3, "Dead": 4, "Epithelial": 5 }

    for i in tqdm(range(len(masks))):
        img_name = "{}.jpg".format(i)

        # 是每幅图都增加一个image吧
        # if img_name not in json_dict['images'].keys():
        image_id +=1
        height, width=256,256
        image = {'file_name': img_name, 'height': height, 'width': width,
                'id': image_id}
#         print(image)
        json_dict['images'].append(image)

        for c in range(5):
            category_id = c+1
            # class_name = "class_{}".format(c)
            # category_id = classname_to_id[class_name]

            mask = masks[i, :, :, c] # 变成一个二维变量
            
            # 获取全部的 instance_id
            instance_ids = [x for x in np.unique(mask) if x!=0]  # 去除0， 背景 id


            for inst_id in instance_ids: # 对每个 instance 生成一张 mask
                
                
                ma = np.where(mask == inst_id, 1, 0)
                # 保存为图片
                # mask_save_path = os.path.join(mask_save_base, "{}.jpg".format(i))
                # cv2.imwrite(mask_save_path, ma)

                # 直接处理，转换为 coco seg？

                # 首先使用 ma 获取bbox
                [x, y, w, h] = find_bbox(ma)[0]

                # 使用 ma 获取 segementation
                seg = mask2polygon(ma)
                if len(seg) == 0:
                    continue
                ann = {'area': w*h, 'iscrowd':0, 'image_id': image_id,
                        'bbox': [x, y, w, h],
                       'category_id': category_id, 
                       'id': bnd_id, 
                       'ignore': 0,
                       'segmentation':seg}
                
                bnd_id = bnd_id + 1
                json_dict['annotations'].append(ann)
        # break

    for cate, cid in classname_to_id.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    
    json_fp = open(json_file, 'w',encoding='utf-8')
    json_str = json.dumps(json_dict,cls=NpEncoder)
    json_fp.write(json_str)
    json_fp.close()

def transfer2img(img_path, base_path):
    imgs = np.load(img_path)
    for i, img in tqdm(enumerate(imgs)):
        image_path = os.path.join(base_path, "{}.jpg".format(i))
        cv2.imwrite(image_path, cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR))



if __name__ == "__main__":
    # 输入格式是？ 假如是针对一个文件夹中的全部文件进行转换
    input_dir_base = '/root/autodl-tmp/datasets/pannuke/pn_format'
    output_dir_base = '/root/autodl-tmp/datasets/pannuke/coco_format'
    mk_file(output_dir_base)
    map_dict = {'1': 'train', '2': 'val', '3':'test'}
    for i in range(3):
        i = str(i+1)
        name = map_dict[str(i)]
    
        # 输入地址 images, masks
        img_path = '{}/{}/images.npy'.format(input_dir_base, map_dict[str(i)])
        masks_path = '{}/{}/masks.npy'.format(input_dir_base, map_dict[str(i)])

        # 输出地址， images
        img_path_base = '{}/images/{}'.format(output_dir_base, name)
        mk_file(img_path_base)

        base_dir = "{}/annotations".format(output_dir_base, name)
        mk_file(base_dir)
        json_file = "{}/instance_{}.json".format(base_dir, name)

        # 生成imgs
        transfer2img(img_path, img_path_base)
        transfer2coco(img_path_base, masks_path, json_file)
        # break
#     img_path_base = "/root/autodl-tmp/mmdet_project/dataset/pannuke/training_data/images/train"
#     masks = np.load("/root/autodl-tmp/mmdet_project/dataset/pannuke/raw/fold_1/masks.npy")
#     # masks = np.load("/root/autodl-tmp/mmdet_project/dataset/pannuke/raw/masks_sample.npy") # for test

#     json_dict = {"images": [], "type": "instances", "annotations": [],
#                  "categories": []}
#     bnd_id = 0
#     image_id= 0
    
#     # category id 要从0开始吗？
#     classname_to_id = {"background": 0, "class_0": 1, "class_1": 2, "class_2": 3, "class_3": 4, "class_4": 5}

    
#     for i in tqdm(range(len(masks))):
#         img_name = "{}.jpg".format(i)

#         # 是每幅图都增加一个image吧
#         # if img_name not in json_dict['images'].keys():
#         image_id +=1
#         height, width=256,256
#         image = {'file_name': img_name, 'height': height, 'width': width,
#                 'id': image_id}
# #         print(image)
#         json_dict['images'].append(image)

#         for c in range(5):
#             class_name = "class_{}".format(c)
#             category_id = classname_to_id[class_name]

#             mask = masks[i, :, :, c] # 变成一个二维变量
            
#             # 获取全部的 instance_id
#             instance_ids = [x for x in np.unique(mask) if x!=0]  # 去除0， 背景 id


#             for inst_id in instance_ids: # 对每个 instance 生成一张 mask
#                 bnd_id = bnd_id + 1
                
#                 ma = np.where(mask == inst_id, 1, 0)
#                 # 保存为图片
#                 # mask_save_path = os.path.join(mask_save_base, "{}.jpg".format(i))
#                 # cv2.imwrite(mask_save_path, ma)

#                 # 直接处理，转换为 coco seg？

#                 # 首先使用 ma 获取bbox
#                 [x, y, w, h] = find_bbox(ma)[0]

#                 # 使用 ma 获取 segementation
#                 seg = mask2polygon(ma)
#                 ann = {'area': w*h, 'iscrowd':0, 'image_id': image_id,
#                         'bbox': [x, y, w, h],
#                        'category_id': category_id, 
#                        'id': bnd_id, 
#                        'ignore': 0,
#                        'segmentation':seg}
                
#                 json_dict['annotations'].append(ann)
                
#     for cate, cid in classname_to_id.items():
#         cat = {'supercategory': 'none', 'id': cid, 'name': cate}
#         json_dict['categories'].append(cat)
    
#     json_file = "/root/autodl-tmp/mmdet_project/dataset/pannuke/training_data/annotations/instance_train.json"
#     json_fp = open(json_file, 'w',encoding='utf-8')
#     json_str = json.dumps(json_dict,cls=NpEncoder)
#     json_fp.write(json_str)
#     json_fp.close()
                

