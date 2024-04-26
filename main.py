# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.chdir('')
import io
import sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import torch_utils as tu
from torch_utils.set_seed import set_seed
import torch
from exp_.exp import EXP

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

'''Other parameters are in the config.py'''
def add_config(parser):
    parser.add_argument('--exp_name', default='deep_learning', choices=['deep_learning'])
    parser.add_argument('--train', default=True,type=str2bool,choices=[True,False], help='Training or not')
    parser.add_argument('--resume', default=False, type=str2bool, choices=[True, False], help='whether to read pre-trained weights')
    parser.add_argument('--output_dir',type=str,default=None,help='if =None will automatically +1 from the existing output file, if specified it will overwrite the existing output file.')
    parser.add_argument('--resume_dir', type=str,default=None,help='Retrieve the location of the checkpoint')
    parser.add_argument('--dp_mode', type=str2bool, default=False,help='Does it run on multiple CUDA')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    # model settings
    parser.add_argument('--model_name', type=str, default='PA2GCN',help=['PA2GCN'])
    parser.add_argument('--data_name', type=str, default='METR-LA', choices=['PeMS-Bay','METR-LA','PEMS04','PEMS08'])

    # dataset settings
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=12)
    # 与模型参数有关的设定
    parser.add_argument('--num_features',help='The feature dimensions (which will be automatically specified based on the dataset)')
    parser.add_argument('--time_features',help='The time steps of input series(which is automatically specified based on the dataset)')
    parser.add_argument('--num_nodes', help='The number of model nodes (which is automatically determined based on the characteristics of the dataset)')
    parser.add_argument('--d_model',type=int,default=64,help='Hidden Layer Dimension 1')
    parser.add_argument('--d_ff',type=int,default=128, help='Hidden Layer Dimension 2')
    parser.add_argument('--num_gcn', type=int, default=10, help='Number of heads in Adaptive Multi-Head GCN')
    parser.add_argument('--patch_len', type=int, default=3, help='Length of a patch')
    parser.add_argument('--stride', type=int, default=1, help='Length of stride')

    parser.add_argument('--points_per_hour', type=int, default=12,help='How many data sampling points in an hour (related to the dataset)')

    parser.add_argument('--info', type=str, default=None, help='Experimental information')
    return parser

def preprocess_args(args):
    args.pin_memory = False
    return args

if __name__ == '__main__':
    args = tu.config.get_args(add_config)
    args = preprocess_args(args)
    set_seed(args.seed)

    print(f"|{'=' * 101}|")
    for key, value in args.__dict__.items():
        print(f"|{str(key):>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")
    print(device)
    if args.exp_name=='deep_learning':
        exp=EXP(args)
    else:
        raise print('There is no exp file with name {0}.'.format(args.exp_name))

    if args.train:
        exp.train()
    with torch.no_grad():
        exp.test()


