'''
Author: your name
Date: 2021-08-27 10:07:50
LastEditTime: 2021-08-27 14:53:16
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ADS_B_select_files_/MCD/svhn2mnist.py
'''
import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse
from ResNet import resnet50

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.resnet = resnet50(1,34)
        # self.fc1 = nn.Linear(8192, 3072)
        # self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = self.resnet(x)
        # print(x.size())
        # x = x.view(x.size(0), 8192)
        # x = F.relu(self.bn1_fc(self.fc1(x)))
        # x = F.dropout(x, training=self.training)
        return x



class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.bn1_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 34)
        # self.bn_fc3 = nn.BatchNorm1d(34)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = self.fc2(x)
        return x
