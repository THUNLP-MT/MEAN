#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse

import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed

from trainer import TrainConfig
from data.pdb_utils import VOCAB



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

    # model
    parser.add_argument('--model', type=str, choices=['refinegnn', 'mcatt', 'mcegnn', \
                        'seq2seq', 'mcatt_noet', 'mcatt_nogl', 'mcatt_nocenter', \
                        'effmcatt', 'effpuremcatt', 'effmcatt_noet', 'effmcatt_nogl', \
                        'effmcegnn'],
                        required=True, help='Model type')
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


def prepare_refine_gnn(args):
    from trainer import RefineGNNTrainer
    from data import AACDataset
    from models.RefineGNN import HierarchicalDecoder
    from models.RefineGNN.utils import RefineGNNConfig

    ########### load your train / valid set ###########
    train_set = AACDataset(args.train_set)
    train_set.mode = args.mode
    train_set.rearrange_by(args.cdr_type, args.batch_size)
    valid_set = AACDataset(args.valid_set)
    valid_set.mode = args.mode
    valid_set.rearrange_by(args.cdr_type, args.batch_size)

    ########## set your collate_fn ##########
    _collate_fn = AACDataset.collate_fn

    ########## define your model #########
    model = HierarchicalDecoder(RefineGNNConfig(
        vocab_size=len(VOCAB), cdr_type=args.cdr_type)
    )
    return RefineGNNTrainer, train_set, valid_set, _collate_fn, model


def prepare_seq2seq(args):
    from trainer import Seq2SeqTrainer
    from data import AACSeqDataset
    from models.Seq2Seq import Seq2Seq
    from models.Seq2Seq.utils import Seq2SeqConfig

    ########### load your train / valid set ###########
    train_set = AACSeqDataset(args.train_set)
    train_set.mode = args.mode
    valid_set = AACSeqDataset(args.valid_set)
    valid_set.mode = args.mode

    ########## set your collate_fn ##########
    _collate_fn = AACSeqDataset.collate_fn

    ########## define your model #########
    model = Seq2Seq(Seq2SeqConfig(
        vocab_size=len(VOCAB), cdr_type=args.cdr_type)
    )
    return Seq2SeqTrainer, train_set, valid_set, _collate_fn, model


def prepare_efficient_mc_att(args):
    from trainer import MCAttTrainer
    from data import EquiAACDataset
    from models.MCAttGNN import EfficientMCAttModel

    ########### load your train / valid set ###########
    train_set = EquiAACDataset(args.train_set)
    train_set.mode = args.mode
    train_set.has_center = False
    valid_set = EquiAACDataset(args.valid_set)
    valid_set.mode = args.mode
    valid_set.has_center = False

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


def prepare_efficient_pure_mc_att(args):
    from trainer import MCAttTrainer
    from data import EquiAACDataset
    from models.MCAttGNN import EfficientPureMCAttModel

    ########### load your train / valid set ###########
    train_set = EquiAACDataset(args.train_set)
    train_set.mode = args.mode
    train_set.has_center = False
    valid_set = EquiAACDataset(args.valid_set)
    valid_set.mode = args.mode
    valid_set.has_center = False

    ########## set your collate_fn ##########
    _collate_fn = EquiAACDataset.collate_fn

    ########## define your model #########
    n_channel = valid_set[0]['X'].shape[1]
    model = EfficientPureMCAttModel(
        args.embed_size, args.hidden_size, n_channel, n_edge_feats=1,
        n_layers=args.n_layers, cdr_type=args.cdr_type, alpha=args.alpha
    )
    return MCAttTrainer, train_set, valid_set, _collate_fn, model


def prepare_mc_att(args):
    from trainer import MCAttTrainer
    from data import EquiAACDataset
    from models.MCAttGNN import MCAttModel

    ########### load your train / valid set ###########
    train_set = EquiAACDataset(args.train_set)
    train_set.mode = args.mode
    valid_set = EquiAACDataset(args.valid_set)
    valid_set.mode = args.mode

    ########## set your collate_fn ##########
    _collate_fn = EquiAACDataset.collate_fn

    ########## define your model #########
    n_channel = valid_set[0]['X'].shape[1]
    model = MCAttModel(
        args.embed_size, args.hidden_size, n_channel, n_edge_feats=1,
        n_layers=args.n_layers, cdr_type=args.cdr_type, alpha=args.alpha
    )
    return MCAttTrainer, train_set, valid_set, _collate_fn, model


# for ablation
def prepare_mc_egnn(args):
    from trainer import MCAttTrainer
    from data import EquiAACDataset
    from models.MCAttGNN import MCEGNNModel

    ########### load your train / valid set ###########
    train_set = EquiAACDataset(args.train_set)
    train_set.mode = args.mode
    valid_set = EquiAACDataset(args.valid_set)
    valid_set.mode = args.mode

    ########## set your collate_fn ##########
    _collate_fn = EquiAACDataset.collate_fn

    ########## define your model #########
    n_channel = valid_set[0]['X'].shape[1]
    model = MCEGNNModel(
        args.embed_size, args.hidden_size, n_channel, n_edge_feats=1,
        n_layers=args.n_layers, cdr_type=args.cdr_type, alpha=args.alpha
    )
    return MCAttTrainer, train_set, valid_set, _collate_fn, model


def prepare_mc_att_noet(args):
    from trainer import MCAttTrainer
    from data import EquiAACDataset
    from models.MCAttGNN import MCAttModel

    ########### load your train / valid set ###########
    print_log('No edge feature')
    train_set = EquiAACDataset(args.train_set)
    train_set.mode = args.mode
    train_set.has_edge_type = False
    valid_set = EquiAACDataset(args.valid_set)
    valid_set.mode = args.mode
    valid_set.has_edge_type = False

    ########## set your collate_fn ##########
    _collate_fn = EquiAACDataset.collate_fn

    ########## define your model #########
    n_channel = valid_set[0]['X'].shape[1]
    model = MCAttModel(
        args.embed_size, args.hidden_size, n_channel, n_edge_feats=0,
        n_layers=args.n_layers, cdr_type=args.cdr_type, alpha=args.alpha
    )
    return MCAttTrainer, train_set, valid_set, _collate_fn, model


def prepare_mc_att_nogl(args):
    trainer, train_set, valid_set, _collate_fn, model = prepare_mc_att(args)
    print_log('No global node')
    train_set.has_global_node = False
    valid_set.has_global_node = False

    return trainer, train_set, valid_set, _collate_fn, model


def prepare_mc_att_nocenter(args):
    from trainer import MCAttTrainer
    from data import EquiAACDataset
    from models.MCAttGNN import MCAttModel

    ########### load your train / valid set ###########
    print_log('No side chain feature')
    train_set = EquiAACDataset(args.train_set)
    train_set.mode = args.mode
    train_set.has_center = False
    valid_set = EquiAACDataset(args.valid_set)
    valid_set.mode = args.mode
    valid_set.has_center = False

    ########## set your collate_fn ##########
    _collate_fn = EquiAACDataset.collate_fn

    ########## define your model #########
    n_channel = valid_set[0]['X'].shape[1]
    model = MCAttModel(
        args.embed_size, args.hidden_size, n_channel, n_edge_feats=1,
        n_layers=args.n_layers, cdr_type=args.cdr_type, alpha=args.alpha
    )
    return MCAttTrainer, train_set, valid_set, _collate_fn, model


def prepare_efficient_mc_egnn(args):
    from trainer import MCAttTrainer
    from data import EquiAACDataset
    from models.MCAttGNN import EfficientMCEGNNModel

    ########### load your train / valid set ###########
    train_set = EquiAACDataset(args.train_set)
    train_set.mode = args.mode
    train_set.has_center = False
    valid_set = EquiAACDataset(args.valid_set)
    valid_set.mode = args.mode
    valid_set.has_center = False

    ########## set your collate_fn ##########
    _collate_fn = EquiAACDataset.collate_fn

    ########## define your model #########
    n_channel = valid_set[0]['X'].shape[1]
    model = EfficientMCEGNNModel(
        args.embed_size, args.hidden_size, n_channel, n_edge_feats=1,
        n_layers=args.n_layers, cdr_type=args.cdr_type, alpha=args.alpha
    )
    return MCAttTrainer, train_set, valid_set, _collate_fn, model


def prepare_efficient_mc_att_noet(args):
    from trainer import MCAttTrainer
    from data import EquiAACDataset
    from models.MCAttGNN import EfficientMCAttModel

    ########### load your train / valid set ###########
    train_set = EquiAACDataset(args.train_set)
    train_set.mode = args.mode
    train_set.has_center = False
    train_set.has_edge_type = False
    valid_set = EquiAACDataset(args.valid_set)
    valid_set.mode = args.mode
    valid_set.has_center = False
    valid_set.has_edge_type = False

    ########## set your collate_fn ##########
    _collate_fn = EquiAACDataset.collate_fn

    ########## define your model #########
    n_channel = valid_set[0]['X'].shape[1]
    model = EfficientMCAttModel(
        args.embed_size, args.hidden_size, n_channel, n_edge_feats=0,
        n_layers=args.n_layers, cdr_type=args.cdr_type, alpha=args.alpha,
        n_iter=args.n_iter
    )
    return MCAttTrainer, train_set, valid_set, _collate_fn, model


def prepare_efficient_mc_att_nogl(args):
    trainer, train_set, valid_set, _collate_fn, model = prepare_efficient_mc_att(args)
    print_log('No global node')
    train_set.has_global_node = False
    valid_set.has_global_node = False

    return trainer, train_set, valid_set, _collate_fn, model


def main(args):
    if args.model == 'refinegnn':
        prepare_func = prepare_refine_gnn
    elif args.model == 'effmcatt':
        prepare_func = prepare_efficient_mc_att
    elif args.model == 'effpuremcatt':
        prepare_func = prepare_efficient_pure_mc_att
    elif args.model == 'effmcatt_noet':
        prepare_func = prepare_efficient_mc_att_noet
    elif args.model == 'effmcatt_nogl':
        prepare_func = prepare_efficient_mc_att_nogl
    elif args.model == 'effmcegnn':
        prepare_func = prepare_efficient_mc_egnn
    elif args.model == 'mcatt':
        prepare_func = prepare_mc_att
    elif args.model == 'mcatt_noet':
        prepare_func = prepare_mc_att_noet
    elif args.model == 'mcatt_nogl':
        prepare_func = prepare_mc_att_nogl
    elif args.model == 'mcatt_nocenter':
        prepare_func = prepare_mc_att_nocenter
    elif args.model == 'mcegnn':
        prepare_func = prepare_mc_egnn
    elif args.model == 'seq2seq':
        prepare_func = prepare_seq2seq
    else:
        raise NotImplementedError(f'model type {args.model} not implemented')
    Trainer, train_set, valid_set, _collate_fn, model = prepare_func(args)
    if args.local_rank == 0 or args.local_rank == -1:
        print_log(str(args))
        print_log(f'model type: {args.model}, parameters: {sum([p.numel() for p in model.parameters()])}')  # million

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
