# -*- coding:utf-8 -*-

"""extract_patches.py
    对已有的数据集进行 patch 提取操作。
    针对 mmsegmentation 的处理，这里做些调整
    1、将得到的 patch 保存为image, mask 保存为 png
        load_ann 没看懂，就是mask的处理时怎么进行的？
    2、保存位置的设定，需要人为指定？
    
"""

import re
import glob
import os
import tqdm
import pathlib

import numpy as np

from patch_extractor import PatchExtractor
from utils import rm_n_mkdir

from dataset import get_dataset
from PIL import Image



def extract(dataset_name = 'consep'):
    # 先改成 consep 的数据集
    dataset_config = {"consep":{
        "train": {
            "img": (".png", "/root/autodl-tmp/datasets/consep/Train/Images"),
            "ann": (".mat", "/root/autodl-tmp/datasets/consep/Train/Labels"),
        },
        "test": {
            "img": (".png", "/root/autodl-tmp/datasets/consep/Test/Images"),
            "ann": (".mat", "/root/autodl-tmp/datasets/consep/Test/Labels"),
        }},

        "kumar":{
        "train": {
            "img": (".tif", "/root/autodl-tmp/datasets/kumar/train/Images"),
            "ann": (".mat", "/root/autodl-tmp/datasets/kumar/train/Labels"),
        },
        "test": {
            "img": (".tif", "/root/autodl-tmp/datasets/kumar/test_same/Images"),
            "ann": (".mat", "/root/autodl-tmp/datasets/kumar/test_same/Labels"),
        }},

        "cpm15":{
        "train": {
            "img": (".png", "/root/autodl-tmp/datasets/cpm15/train/Images"),
            "ann": (".mat", "/root/autodl-tmp/datasets/cpm15/train/Labels"),
        },
        "test": {
            "img": (".png", "/root/autodl-tmp/datasets/cpm15/test/Images"),
            "ann": (".mat", "/root/autodl-tmp/datasets/cpm15/test/Labels"),
        }},
    
        "cpm17":{
        "train": {
            "img": (".png", "/root/autodl-tmp/datasets/cpm17/train/Images"),
            "ann": (".mat", "/root/autodl-tmp/datasets/cpm17/train/Labels"),
        },
        "test": {
            "img": (".png", "/root/autodl-tmp/datasets/cpm17/test/Images"),
            "ann": (".mat", "/root/autodl-tmp/datasets/cpm17/test/Labels"),
        }},
    }
    dataset_info = dataset_config[dataset_name]


    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = False

    win_size = [540, 540]
    step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders.
                            # 'valid'- only extract from valid regions.

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)

    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        # out_dir = "%s/%s/%dx%d_%dx%d/" % (
        #     save_root,
        #     # dataset_name,
        #     split_name,
        #     win_size[0],
        #     win_size[1],
        #     step_size[0],
        #     step_size[1],
        # )
        img_out_dir = "/root/autodl-tmp/datasets/{}/images/{}".format(dataset_name, split_name)
        label_out_dir = "/root/autodl-tmp/datasets/{}/labels/{}".format(dataset_name, split_name)
        label_inst_dir = "/root/autodl-tmp/datasets/{}/labels_inst/{}".format(dataset_name, split_name)
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(img_out_dir) # 这里我改成了保留已存在的文件夹，而不是直接删除
        rm_n_mkdir(label_out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                # 这里的 patch 是不是 [256,256, 5] 呢？如果是的话，就在这里处理掉。
                # 这里是保存为 npy，转换为保存 png
                # image = patch[:,:,:3]
                # image_filename = "{0}/{1:03d}.jpg".format(img_out_dir, idx)
                # Image.fromarray(image.astype(np.uint8)).save(image_filename, 'JPEG')
                # label = patch[:,:,-1]
                # label_filename = "{0}/{1:03d}.png".format(label_out_dir, idx)
                # Image.fromarray(label.astype(np.uint8)).save(label_filename, 'PNG')
                # np.save("{0}/{1}_{2:03d}.npy".format(label_out_dir, base_name, idx), patch)
                
                ## 如果只有type的话，无法转换成coco格式做instance seg
                ### 取后两维保存，一个是 inst, 一个是 type
                np.save("{0}/{1:03d}.npy".format(label_inst_dir, idx), patch[:,:,-2:])
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()
# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # # Determines whether to extract type map (only applicable to datasets with class labels).
    # type_classification = True

    # win_size = [540, 540]
    # step_size = [164, 164]
    # extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders.
    #                         # 'valid'- only extract from valid regions.

    # # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # # This used to get the specific dataset img and ann loading scheme from dataset.py
    # dataset_name = "consep"
    # save_root = "dataset/training_data/%s/" % dataset_name

    # # a dictionary to specify where the dataset path should be
    # # 这应该是原始数据吧
    # dataset_info = {
    #     "train": {
    #         "img": (".png", "F:/viax/dataset/hovernet/data/consep/Train/Images/"),
    #         "ann": (".mat", "F:/viax/dataset/hovernet/data/consep/Train/Labels/"),
    #     },
    #     "valid": {
    #         "img": (".png", "F:/viax/dataset/hovernet/data/consep/Test/Images/"),
    #         "ann": (".mat", "F:/viax/dataset/hovernet/data/consep/Test/Labels/"),
    #     },
    # }

    # patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    # parser = get_dataset(dataset_name)
    # xtractor = PatchExtractor(win_size, step_size)
    # for split_name, split_desc in dataset_info.items():
    #     img_ext, img_dir = split_desc["img"]
    #     ann_ext, ann_dir = split_desc["ann"]

    #     out_dir = "%s/%s/%dx%d_%dx%d/" % (
    #         save_root,
    #         # dataset_name,
    #         split_name,
    #         win_size[0],
    #         win_size[1],
    #         step_size[0],
    #         step_size[1],
    #     )
    #     file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
    #     file_list.sort()  # ensure same ordering across platform

    #     rm_n_mkdir(out_dir)

    #     pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    #     pbarx = tqdm.tqdm(
    #         total=len(file_list), bar_format=pbar_format, ascii=True, position=0
    #     )

    #     for file_idx, file_path in enumerate(file_list):
    #         base_name = pathlib.Path(file_path).stem

    #         img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
    #         ann = parser.load_ann(
    #             "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
    #         )

    #         # *
    #         img = np.concatenate([img, ann], axis=-1)
    #         sub_patches = xtractor.extract(img, extract_type)

    #         pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    #         pbar = tqdm.tqdm(
    #             total=len(sub_patches),
    #             leave=False,
    #             bar_format=pbar_format,
    #             ascii=True,
    #             position=1,
    #         )

    #         for idx, patch in enumerate(sub_patches):
    #             np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
    #             pbar.update()
    #         pbar.close()
    #         # *

    #         pbarx.update()
    #     pbarx.close()
    extract('consep')










