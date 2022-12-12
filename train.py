#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse

import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed

from trainer import TrainConfig


def parse():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='111', help='H/L/Antigen, 1 for include, 0 for exclude')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use in training')
    parser.add_argument('--early_stop', action='store_true', help='Whether to use early stop')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    ## shared
    parser.add_argument('--cdr_type', type=str, default='3', help='type of cdr')
    ## for Multi-Channel Attetion model
    parser.add_argument('--embed_size', type=int, default=64, help='embed size of amino acids')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--alpha', type=float, default=0.05, help='scale mse loss of coordinates')
    parser.add_argument('--anneal_base', type=float, default=1, help='Exponential lr decay, 1 for not decay')
    ## for efficient version
    parser.add_argument('--n_iter', type=int, default=5, help='Number of iterations')
    return parser.parse_args()


def prepare_efficient_mc_att(args):
    from trainer import MCAttTrainer
    from data import EquiAACDataset
    from models.MCAttGNN import EfficientMCAttModel

    ########### load your train / valid set ###########
    train_set = EquiAACDataset(args.train_set)
    train_set.mode = args.mode
    valid_set = EquiAACDataset(args.valid_set)
    valid_set.mode = args.mode

    ########## set your collate_fn ##########
    _collate_fn = EquiAACDataset.collate_fn

    ########## define your model #########
    n_channel = valid_set[0]['X'].shape[1]
    model = EfficientMCAttModel(
        args.embed_size, args.hidden_size, n_channel, n_edge_feats=1,
        n_layers=args.n_layers, cdr_type=args.cdr_type, alpha=args.alpha,
        n_iter=args.n_iter
    )
    return MCAttTrainer, train_set, valid_set, _collate_fn, model


def main(args):
    Trainer, train_set, valid_set, _collate_fn, model = prepare_efficient_mc_att(args)
    if args.local_rank == 0 or args.local_rank == -1:
        print_log(str(args))
        print_log(f'parameters: {sum([p.numel() for p in model.parameters()])}')  # million

    if len(args.gpus) > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle, seed=args.seed)
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        train_sampler = None
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=_collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=_collate_fn)
    config = TrainConfig(args, args.save_dir, args.lr, args.max_epoch, grad_clip=args.grad_clip, early_stop=args.early_stop, anneal_base=args.anneal_base)
    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args = parse()
    setup_seed(args.seed)
    main(args)
