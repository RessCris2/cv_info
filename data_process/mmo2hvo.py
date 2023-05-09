# -*- coding:utf-8 -*-
"""先使用 mmo2pn 转换为 pn,再转换为 hvo
"""
# def infer_dir(model, input_dir, output_dir):
import mmo2hvi # import infer_dir
import pn2hvi
import hvi2hvo
from mmdet.apis import init_detector, inference_detector
import mmcv


def mmo2hvo(infer_dir, output_hvi_dir,  output_hover_dir):
    """ 输入是需要预测的文件夹, coco_format
        输出是预测结果的 hover mat
    """

    config_file = '/root/autodl-tmp/work_dirs/mask_rcnn_r50_fpn_1x_pannuke/mask_rcnn_r50_fpn_1x_pannuke.py'
    checkpoint_file = '/root/autodl-tmp/work_dirs/mask_rcnn_r50_fpn_1x_pannuke/latest.pth'
    # 根据配置文件和 checkpoint 文件构建模型
    # model = init_detector(config_file, checkpoint_file, device='cuda:0')

    ## 先将预测结果放进 output_hvi_dir
    # mmo2hvi.infer_dir(model, infer_dir, output_hvi_dir)

    ## 再转换
    hvi2hvo.turn_mat(output_hvi_dir, output_hover_dir)


if __name__ == "__main__":
    infer_dir = "/root/autodl-tmp/datasets/pannuke/coco_format/images/test"  # 这里一定要注意，infer 的文件夹是 test
    output_pn_dir =  '/root/autodl-tmp/datasets/pn_maskrcnn_infer/20230324/pn_format'
    output_hvi_dir = '/root/autodl-tmp/datasets/pn_maskrcnn_infer/20230324/hvi_format'
    output_hover_dir = '/root/autodl-tmp/datasets/pn_maskrcnn_infer/20230324/hvo_format'
    mmo2hvo(infer_dir, output_hvi_dir,  output_hover_dir)