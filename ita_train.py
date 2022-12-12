#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import argparse
from time import time
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ITAWrapper, AAComplex
from trainer.abs_trainer import TrainConfig
from generate import set_cdr
from evaluation.rmsd import compute_rmsd, kabsch
from evaluation import pred_ddg
from utils.logger import print_log
from utils.random_seed import setup_seed


def prepare_efficient_mc_att(model, mode, data_path, batch_size):
    from trainer import MCAttTrainer
    from data import EquiAACDataset

    dataset = EquiAACDataset(data_path)
    dataset.mode = mode
    return dataset, MCAttTrainer

def get_config(ckpt):
    directory = os.path.split(ckpt)[0]
    directory = os.path.split(directory)[0]
    config = os.path.join(directory, 'train_config.json')
    with open(config, 'r') as fin:
        config = json.load(fin)['args']
    # model_type = re.search(r'model=\'(.*?)\'', config).group(1)
    mode = re.search(r'mode=\'([0-1][0-1][0-1])\'', config).group(1)
    return mode


def parse():
    parser = argparse.ArgumentParser(description='ITA training')
    parser.add_argument('--pretrain_ckpt', type=str, required=True,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--test_set', type=str, required=True,
                        help='Path to test set (antibodies to be optimized)')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples each iteration')
    parser.add_argument('--n_tries', type=int, default=50, help='Number of tries each iteration')
    parser.add_argument('--n_iter', type=int, default=20, help='Number of iterations to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU to use, -1 for cpu')
    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=1, help='number of epochs per iteration')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--update_freq', type=int, default=1, help='Model update frequency. if not 1, true batch size will be freq * batch_size')
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def valid_check(seq):
    charge_valid, motif_valid, seq_valid = True, True, True
    # charge
    charge = 0
    for res in seq:
        if res == 'R' or res == 'K':
            charge += 1
        elif res == 'H':
            charge += 0.1
        elif res == 'D' or res == 'E':
            charge -= 1
    if charge < -2.0 or charge > 2.0:
        charge_valid = False

    # motif
    for i in range(len(seq) - 2):
        motif = seq[i:i+3]
        if motif[0] == 'N' and (motif[-1] == 'S' or motif[-1] == 'T'):
            motif_valid = False
            break
        
    # seq
    longest, previous, cnt = 0, None, 0
    for res in seq:
        if res == previous:
            cnt += 1
            longest = max(longest, cnt)
        else:
            cnt = 1
        previous = res
    if longest > 5:
        seq_valid = False
    
    return motif_valid and charge_valid and seq_valid
    

def main(args):
    print(str(args))
    mode = get_config(args.pretrain_ckpt)
    print(f'mode: {mode}')
    model = torch.load(args.pretrain_ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    dataset, Trainer = prepare_efficient_mc_att(model, mode, args.test_set, args.batch_size)
    itawrapper = ITAWrapper(dataset, args.n_samples)
    origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]
    
    valid_loader = DataLoader(dataset, batch_size=args.batch_size * args.update_freq,
                              num_workers=args.num_workers,
                              shuffle=False,
                              collate_fn=dataset.collate_fn)
    config = TrainConfig(args, args.save_dir, args.lr, args.epoch, grad_clip=args.grad_clip)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'train_config.json'), 'w') as fout:
        json.dump(config.__dict__, fout)

    def fake_log(*args, **kwargs):
        return

    # writing original structrues
    origin_cplx_paths = []
    out_dir = os.path.join(args.save_dir, 'original')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print_log(f'Writing original structures to {out_dir}')
    for cplx in tqdm(origin_cplx):
        pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
        cplx.to_pdb(pdb_path)
        origin_cplx_paths.append(os.path.abspath(pdb_path))
    log = open(os.path.join(args.save_dir, 'log.txt'), 'w')
    best_round, best_score = -1, 1e10

    for r in range(args.n_iter):
        start = time()
        res_dir = os.path.join(args.save_dir, f'iter_{r}')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        # generate better samples
        print_log('Generating samples')
        model.eval()
        scores = []
        for i in tqdm(range(len(dataset))):
            origin_input = dataset[i]
            inputs = [origin_input for _ in range(args.n_tries)]
            candidates, results = [], []
            with torch.no_grad():
                batch = dataset.collate_fn(inputs)
                ppls, seqs, xs, true_xs, aligned = model.infer(batch, device, greedy=False)
                results.extend([(ppls[i], seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])
            recorded, candidate_pool = {}, []
            for n, (ppl, seq, x, true_x, aligned) in enumerate(results):
                if seq in recorded:
                    continue
                recorded[seq] = True
                if ppl > 10:
                    # print_log(f'High PPL {ppl}, skip')
                    continue
                if not valid_check(seq):
                    # print_log(f'Validity check failed, skip')
                    continue
                if not aligned:
                    ca_aligned, rotation, t = kabsch(x[:, 1, :], true_x[:, 1, :])
                    x = np.dot(x - np.mean(x, axis=0), rotation) + t
                candidate_pool.append((ppl, seq, x, n))
            sorted_cand_idx = sorted([j for j in range(len(candidate_pool))], key=lambda j: candidate_pool[j][0])
            for j in sorted_cand_idx:
                ppl, seq, x, n = candidate_pool[j]
                new_cplx = set_cdr(origin_cplx[i], seq, x, cdr='H' + str(model.cdr_type))
                pdb_path = os.path.join(res_dir, new_cplx.get_id() + f'_{n}.pdb')
                new_cplx.to_pdb(pdb_path)
                new_cplx = AAComplex(
                    new_cplx.pdb_id, new_cplx.peptides,
                    new_cplx.heavy_chain, new_cplx.light_chain,
                    new_cplx.antigen_chains)
                try:
                    score = pred_ddg(origin_cplx_paths[i], os.path.abspath(pdb_path))
                except Exception as e:
                    print_log(f'ddg prediction failed: {str(e)}', level='ERROR')
                    score = 0
                if score < 0:
                    candidates.append((new_cplx, score))
                    scores.append(score)
                if len(candidates) >= args.n_samples:
                    break
            while len(candidates) < args.n_samples:
                candidates.append((origin_cplx[i], 0))
                scores.append(0)
            itawrapper.update_candidates(i, candidates)

        itawrapper.finish_update()

        mean_score = np.mean(scores)
        if mean_score < best_score:
            best_round, best_score = r - 1, mean_score
        log_line = f'model from iteration {r - 1}, ddg mean {mean_score}, std {np.std(scores)}, history best {best_score} at round {best_round}'
        print_log(log_line)
        log.write(log_line + '\n')

        # train
        print_log(f'Iteration {r}, result directory: {res_dir}')
        print_log(f'Start training')
        model.train()
        train_loader = DataLoader(itawrapper, batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  collate_fn=itawrapper.collate_fn)
        trainer = Trainer(model, train_loader, valid_loader, config)
        trainer.log = fake_log
        optimizer = trainer.get_optimizer()
        batch_idx = 0
        for e in range(args.epoch):
            for batch in train_loader:
                batch = trainer.to_device(batch, device)
                loss = trainer.train_step(batch, batch_idx) / args.update_freq
                loss.backward()
                batch_idx += 1
                if batch_idx % args.update_freq == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
        if batch_idx % args.update_freq != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        # save model
        model_path = os.path.join(args.save_dir, f'iter_{r}.ckpt')
        print_log(f'Saving to {model_path}')
        torch.save(model, model_path)

        print_log(f'Elapsed: {time() - start} s')

    log.close()


if __name__ == '__main__':
    args = parse()
    setup_seed(args.seed)
    main(args)
