#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=50, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")##参与每轮训练的客户端比例
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    # 添加谱聚类参数
    parser.add_argument('--sigma', type=float, default=0.4,
                        help="sigma parameter for spectral clustering")
    parser.add_argument('--epsilon', type=float, default=None,
                        help="epsilon threshold for spectral clustering")
    parser.add_argument('--local_ep_init', type=int, default=1,
                        help="number of local epochs for initial model training")
    # non-iid partitioning parameters
    parser.add_argument('--partition', type=str, default='noniid-labeldir',
                       help="data partition method: homo, noniid-labeldir, noniid-#label1-9, iid-diff-quantity")
    parser.add_argument('--beta', type=float, default=0.1,
                       help="parameter for non-iid data distribution (Dirichlet)")
    parser.add_argument('--data_path', type=str, default='../data/',
                       help="path to save dataset")

    
    # FedRS parameters
    parser.add_argument('--method', type=str, default='fedavg', 
                       help="aggregation method: fedavg, fedrs")
    parser.add_argument('--fedrs_alpha', type=float, default=0.5, 
                       help="hyper parameter for FedRS restricted softmax")
    parser.add_argument('--min_le', type=int, default=1, 
                       help="minimum number of local epochs for FedRS")
    parser.add_argument('--max_le', type=int, default=5, 
                       help="maximum number of local epochs for FedRS")

    # Data partitioning method selection
    parser.add_argument('--use_sampling', action='store_true',
                       help="use sampling.py data partitioning instead of data_partition.py")
    
    # Hierarchical FL parameters
    parser.add_argument('--ES_k2', type=int, default=2,
                       help="number of ES layer aggregation rounds")
    parser.add_argument('--EH_k3', type=int, default=2,
                       help="number of EH layer aggregation rounds")
    parser.add_argument('--num_processes', type=int, default=8,
                       help="number of parallel processes for client training")

    # Data saving and loading parameters
    parser.add_argument('--save_data', action='store_true',
                       help="save client data distribution to file")
    parser.add_argument('--load_data', type=str, default=None,
                       help="load client data distribution from specified file path")
    parser.add_argument('--data_save_dir', type=str, default='./saved_data/',
                       help="directory to save client data distribution files")
    
    args = parser.parse_args()
    # 动态设置num_classes和num_channels
    if args.dataset == 'mnist':
        args.num_classes = 10
        args.num_channels = 1
    elif args.dataset == 'cifar':
        args.num_classes = 10
        args.num_channels = 3
    elif args.dataset == 'cifar100':
        args.num_classes = 100  # CIFAR-100有100个类别
        args.num_channels = 3
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    return args
