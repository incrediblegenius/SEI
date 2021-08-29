'''
Author: your name
Date: 2021-08-25 14:13:08
LastEditTime: 2021-08-27 09:15:47
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ADS_B_select_files_/classification/model/build_gen.py
'''

import usps
import syn2gtrsb
import svhn2mnist

def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return usps.Feature()
    elif source == 'svhn':
        return svhn2mnist.Feature()
    elif source == 'synth':
        return syn2gtrsb.Feature()


def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Predictor()
    if source == 'svhn':
        return svhn2mnist.Predictor()
    if source == 'synth':
        return syn2gtrsb.Predictor()

