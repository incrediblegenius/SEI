'''
Author: your name
Date: 2021-08-25 14:13:08
LastEditTime: 2021-08-27 09:45:03
LastEditors: your name
Description: In User Settings Edit
FilePath: /ADS_B_select_files_/classification/datasets/gtsrb.py
'''
import numpy as np
import pickle as pkl

def load_gtsrb():
    data_target = pkl.load(open('../data/data_gtsrb'))
    target_train = np.random.permutation(len(data_target['image']))
    data_t_im = data_target['image'][target_train[:31367], :, :, :]
    data_t_im_test = data_target['image'][target_train[31367:], :, :, :]
    data_t_label = data_target['label'][target_train[:31367]] + 1
    data_t_label_test = data_target['label'][target_train[31367:]] + 1
    data_t_im = data_t_im.transpose(0, 3, 1, 2).astype(np.float32)
    data_t_im_test = data_t_im_test.transpose(0, 3, 1, 2).astype(np.float32)
    return data_t_im, data_t_label, data_t_im_test, data_t_label_test
