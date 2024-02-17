import os
import os.path as osp
import sys
import shutil
import numpy as np

### Here to change ###
LIST_PATH_DATASET = ['/home/pengqihang/workspace/EchoFusion/data/k-radar']
### Here to change ###

def get_dict_key_seq_val_label(path_datset):
    dict_ret = dict()
    list_seq_num = sorted(os.listdir(path_datset))

    for seq in list_seq_num:
        if seq == 'ImageSets':
            continue
        list_labels_per_seq = sorted(os.listdir(osp.join(path_datset, seq, 'info_label')))
        dict_ret[seq] = list_labels_per_seq
    
    return dict_ret

def sort_list_char_via_int_order(list_sort):
    ''''
    * e.g., in  : ['9', '10', '1', '15', '8', '100']
    *       out : ['1', '8', '9', '10', '15', '100']
    '''
    list_ret = sorted(list(map(lambda x: int(x), list_sort)))
    list_ret = list(map(lambda x: f'{x}', list_ret))

    return list_ret

def add_seq_and_list_label_to_txt(txt, seq, list_label):
    txt_ret = txt
    for label in list_label:
        txt_ret += f'{seq},{label}\n'
    return txt_ret

def divide_to_consecutive_quater(list_to_be):
    num_val = len(list_to_be)

    idx_0 = int(num_val*0.20)
    idx_1 = int(num_val*0.25)
    idx_2 = int(num_val*0.50)
    idx_3 = int(num_val*0.70)
    idx_4 = int(num_val*0.75)

    return list_to_be[:idx_0], list_to_be[idx_0:idx_1], \
            list_to_be[idx_2:idx_3], list_to_be[idx_3:idx_4]

def get_txt_train_and_val(list_path_dataset):
    dict_seq = dict()
    for path_dataset in list_path_dataset:
        dict_seq.update(get_dict_key_seq_val_label(path_dataset))
    
    sorted_keys = sort_list_char_via_int_order(list(dict_seq.keys()))
    
    txt_train = ''
    txt_val = ''

    print(f'sequences = {sorted_keys}')
    for seq in sorted_keys:
        list_0, list_1, list_2, list_3 = divide_to_consecutive_quater(dict_seq[seq])
        txt_train = add_seq_and_list_label_to_txt(txt_train, seq, list_0)
        txt_val  = add_seq_and_list_label_to_txt(txt_val,  seq, list_1)
        txt_train = add_seq_and_list_label_to_txt(txt_train, seq, list_2)
        txt_val  = add_seq_and_list_label_to_txt(txt_val,  seq, list_3)
    
    return txt_train[:-1], txt_val[:-1] # delete '\n'

def save_to_path(txt, path_file):
    f = open(path_file, 'w')
    f.write(txt)
    f.close()

if __name__ == '__main__':
    txt_train, txt_val = get_txt_train_and_val(LIST_PATH_DATASET)
    save_to_path(txt_train, './data/k-radar/ImageSets/train.txt')
    save_to_path(txt_val, './data/k-radar/ImageSets/val.txt')
