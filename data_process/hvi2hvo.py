# -*- coding:utf-8 -*-
""" 
    将 hv input format 转换为hv output format
    也就是 【img, inst, type] --> mat 
    
"""

# -*- coding:utf-8 -*-
import sys
sys.path.append("/root/autodl-tmp/hover_net")
import numpy as np
from tqdm import tqdm
import os
import cv2
import scipy.io as sio
import pathlib
import glob
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)

from skimage.segmentation import watershed
#  remove_small_objects



import warnings


def noop(*args, **kargs):
    pass


warnings.warn = noop
# from models.hovernet.post_proc import __proc_np_hv
# from misc.utils import get_bounding_box
def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel. 
    
    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

####
def __proc_np_hv(pred):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred
####
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def turn_mat(input_dir, output_dir):
    file_path = os.path.join(input_dir, "*.npy" )
    file_list = glob.glob(file_path)
    file_list.sort()  # ensure same order [1]
    for file in tqdm(file_list):
        turn_one_mat(file, output_dir)


def turn_one_mat(data_path, output_dir):
    base_name = pathlib.Path(data_path).stem
    pred_inst = np.load(data_path)[:, :, 3]   # 第 4 维是inst
    pred_inst = np.squeeze(pred_inst)
    # pred_inst = __proc_np_hv(pred_inst)
    inst_id_list = np.unique(pred_inst)[1:]  # exlcude background


    pred_type = np.load(data_path)[:, :, -1] # 最后一维是type


    # 去找一个什么东西。。？
    # inst_info_dict = None
    inst_info_dict = {}
    for inst_id in inst_id_list:
        inst_map = pred_inst == inst_id  ## 获取每个inst_id 对应的 mask
        # TODO: chane format of bbox output
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        inst_map = inst_map[
            inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
        ]
        inst_map = inst_map.astype(np.uint8)
        inst_moment = cv2.moments(inst_map)
        inst_contour = cv2.findContours(
            inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # * opencv protocol format may break
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
        # < 3 points dont make a contour, so skip, likely artifact too
        # as the contours obtained via approximation => too small or sthg
        if inst_contour.shape[0] < 3:
            continue
        if len(inst_contour.shape) != 2:
            continue # ! check for trickery shape
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)
        inst_contour[:, 0] += inst_bbox[0][1]  # X
        inst_contour[:, 1] += inst_bbox[0][0]  # Y
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y
        inst_info_dict[inst_id] = {  # inst_id should start at 1
            "bbox": inst_bbox,
            "centroid": inst_centroid,
            "contour": inst_contour,
            "type_prob": None,
            "type": None,
        }


    ## 处理 type 字段
    for inst_id in list(inst_info_dict.keys()):
        rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
        inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
        inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
        inst_map_crop = (
            inst_map_crop == inst_id
        )  # TODO: duplicated operation, may be expensive
        inst_type = inst_type_crop[inst_map_crop]
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        type_dict = {v[0]: v[1] for v in type_list}
        type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
        inst_info_dict[inst_id]["type"] = int(inst_type)
        inst_info_dict[inst_id]["type_prob"] = 1



    # print(inst_info_dict)
    # 然后转换为 mat？
    nuc_val_list = list(inst_info_dict.values())
    # need singleton to make matlab happy
    nuc_uid_list = np.array(list(inst_info_dict.keys()))[:,None]
    nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:,None]
    nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])

    mat_dict = {
                "inst_map" : pred_inst,
                "inst_uid" : nuc_uid_list,
                "inst_type": nuc_type_list,
                "inst_centroid": nuc_coms_list
            }
    save_path = "%s/%s.mat" % (output_dir, base_name)
    sio.savemat(save_path, mat_dict)

if __name__ == '__main__':
    # data_path = "/root/autodl-tmp/hover_net/dataset/training_data/pannuke/test_sample/1.npy"

    # input_dir = "/root/autodl-tmp/hover_net/dataset/training_data/pannuke/test_sample/"
    # output_dir = "/root/autodl-tmp/hover_net/dataset/training_data/pannuke/test_sample_mat"
    # # turn_one_mat(data_path, output_dir)
    # turn_mat(input_dir, output_dir)
    # print('xxx')

    # 读取现有的 pannuke 转换前的格式; 我怀疑是 6 个channel 的 mask [256, 256, 6]
    # data_path = "/root/autodl-tmp/datasets/pannuke/training_data/hvi_format/train/0.npy"
    # output_dir = "/root/autodl-tmp/datasets/pannuke/training_data/hvo_format/"
    # turn_one_mat(data_path, output_dir)
    # # res = np.load(file_path)
    # print('xxx')


    # 将 unet infer 的结果中 hvi_format 转换为 hvo_format
    input_dir = "/root/autodl-tmp/datasets/pn_unet_infer/hvi_format"
    output_dir = "/root/autodl-tmp/datasets/pn_unet_infer/hvo_format"
    turn_mat(input_dir, output_dir)