'''
Author: your name
Date: 2021-08-27 09:40:20
LastEditTime: 2021-08-27 13:48:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ADS_B_select_files_/MCD/main.py
'''
from __future__ import print_function
import argparse
import torch

import os
from solver import Solver

# Training settings
# parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
# parser.add_argument('--all_use', type=str, default='no', metavar='N',
#                     help='use all training data? in usps adaptation')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
#                     help='source only or not')
# parser.add_argument('--eval_only', action='store_true', default=False,
#                     help='evaluation only option')
# parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
#                     help='learning rate (default: 0.0002)')
# parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
#                     help='how many epochs')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--num_k', type=int, default=4, metavar='N',
#                     help='hyper paremeter for generator update')
# parser.add_argument('--one_step', action='store_true', default=False,
#                     help='one step training with gradient reversal layer')
# parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
# parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
#                     help='epoch to resume')
# parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
#                     help='when to restore the model')
# parser.add_argument('--save_model', action='store_true', default=False,
#                     help='save_model or not')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--source', type=str, default='svhn', metavar='N',
#                     help='source dataset')
# parser.add_argument('--target', type=str, default='mnist', metavar='N', help='target dataset')
# parser.add_argument('--use_abs_diff', action='store_true', default=False,
#                     help='use absolute difference value as a measurement')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
# print(args)


def main():
    # if not args.one_step:

    solver = Solver()
    count = 0
    for t in range(30):

        num = solver.train(t)

        count += num
        if t % 1 == 0:
            solver.test(t)

if __name__ == '__main__':
    main()
