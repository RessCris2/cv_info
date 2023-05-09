# -*- coding:utf-8 -*-
""" 
    将 pannuke 转换为hv input format, 
    [img, inst, type] npy 格式
"""

# -*- coding:utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt

## 建立 train, valid ，也及时
def read_file(file_path, save_path):
    images = np.load('{}/images.npy'.format(file_path))
    masks = np.load("{}/masks.npy".format(file_path))

    n = len(images)
    for i in tqdm(range(n)):
        ## 这个地方很诡异
        type_mask = np.argmax(masks[i, :, :, [5, 0, 1, 2, 3, 4]], axis=0)

        # 这里来点测试

        # masks_yan = np.where(masks[i, :, :, :5] > 0, 1, 0 )
        # np.unique(masks_yan.sum(axis=-1)) # 必须只有0或1取值，表示所有维度上最多只有一个值大于0.

        inst_mask = masks[i, :, :, :5].sum(axis=-1, keepdims=1)

        inst_ids = np.unique(inst_mask)[1:]

        ## 对每个维度循环
        t_inst = 0
        mapp = {}
        for c in range(5):
            mask = masks[i, :, :, c]  
            inst_ids = np.unique(mask)[1:]

            for ind, inst_id in enumerate(inst_ids):
                t_inst += 1
                mapp[inst_id] = t_inst
                # 对每个inst重新赋id
                inst_mask = np.where(inst_mask==inst_id, t_inst, inst_mask)
        
        inst_ids = np.unique(inst_mask)[1:]

        inp = np.c_[images[i], inst_mask, type_mask[..., None]]

        save_file_path = os.path.join(save_path, '{}.npy'.format(i))
        np.save(save_file_path, inp)
        # break
 

if __name__ == "__main__":

    # input_dir_base = '/root/autodl-tmp/mmdet_project/dataset/pannuke/raw'
    input_dir_base = "/root/autodl-tmp/datasets/pannuke/training_data/pn_format"
    output_dir_base = '/root/autodl-tmp/datasets/pannuke/training_data/hvi_format'

    map_dict = {'1': 'train', '2': 'val', '3':'test'}
    for i in range(3):
        i = str(i+1)
        name = map_dict[str(i)]
    
        # 输入地址 images, masks
        file_path = '{}/fold_{}'.format(input_dir_base, i)
        # masks_path = '{}/fold_{}/masks.npy'.format(input_dir_base, i)

        #  file_path = "/root/autodl-tmp/hover_net/dataset/raw/pannuke/test"
        save_path = "{}/{}".format(output_dir_base, name)
        read_file(file_path, save_path)
        # break
