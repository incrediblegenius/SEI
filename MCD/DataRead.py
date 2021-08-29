'''
Author: your name
Date: 2021-08-27 10:33:47
LastEditTime: 2021-08-27 11:15:59
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ADS_B_select_files_/MCD/DataLoader.py
'''
from DataSet import DataSet64x64,DataLoader,DataSet16x16,DataLoader
from unaligned_data_loader import UnalignedDataLoader
import h5py
import numpy as np

def data_read(batch_size, path='/home/zsc/ADS_B_select_files_/dataset/Preprocess_64x64_202004xx_to_202011xx.h5' ):
    with h5py.File(path,'r') as f:
        X_train = np.array(f['X_train'][:])
        y_train = np.array(f['y_train'][:])
        X_test = np.array(f['X_test'][:])
        y_test = np.array(f['y_test'][:])
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    S['imgs'] = X_train
    S['labels'] = y_train
    T['imgs'] = X_test
    T['labels'] = y_test

    S_test['imgs'] = X_test
    S_test['labels'] = y_test
    T_test['imgs'] = X_test
    T_test['labels'] = y_test

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size)
    dataset = train_loader.load_data()
    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size)
    dataset_test = test_loader.load_data()

    return dataset, dataset_test