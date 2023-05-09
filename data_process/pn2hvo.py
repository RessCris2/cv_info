# -*- coding:utf-8 -*-
import pn2hvi
import hvi2hvo

def pannuke_pn2hvo(input_dir_base, output_hvo_dir):
    # input_dir_base = "/root/autodl-tmp/datasets/pannuke/pn_format"
    output_dir_base = '/root/autodl-tmp/datasets/pannuke/hvi_format'

    map_dict = {'1': 'train', '2': 'val', '3':'test'}
    for i in range(3):
        i = str(i+1)
        name = map_dict[str(i)]
    
        # 输入地址 images, masks
        file_path = '{}/fold_{}'.format(input_dir_base, i)
        # masks_path = '{}/fold_{}/masks.npy'.format(input_dir_base, i)

        #  file_path = "/root/autodl-tmp/hover_net/dataset/raw/pannuke/test"
        save_path = "{}/{}".format(output_dir_base, name)
        # pn2hvi.read_file(file_path, save_path)
    
        output_hvo_dire = "{}/{}".format(output_hvo_dir, name)
        # 分别转换为 hvo
        hvi2hvo.turn_mat(save_path, output_hvo_dire)
    


if __name__ == "__main__":
    input_dir_base = "/root/autodl-tmp/datasets/pannuke/pn_format"
    output_hvo_dir = '/root/autodl-tmp/datasets/pannuke/hvo_format'
    pannuke_pn2hvo(input_dir_base, output_hvo_dir)