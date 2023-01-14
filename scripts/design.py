#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))
import argparse
from os.path import splitext, basename

import numpy as np
import torch

from data.pdb_utils import AAComplex, Protein, VOCAB
from data.dataset import EquiAACDataset
from generate import set_cdr


def to_tensor(cplx):
    hc, lc = cplx.get_heavy_chain(), cplx.get_light_chain()
    antigen_chains = cplx.get_antigen_chains(interface_only=True, cdr=None)

    # prepare input
    chain_lists, begins = [], []
    chain_lists.append([hc])
    begins.append(VOCAB.BOH)
    chain_lists.append([lc])
    begins.append(VOCAB.BOL)
    chain_lists.append(antigen_chains)
    begins.append(VOCAB.BOA)
    
    X, S, = [], []
    chain_start_ends = []  # tuples of [start, end)

    # format input, box is begin of chain x
    corrupted_idx = []
    for chains, box in zip(chain_lists, begins):
        # judge if the chain has length
        skip = True
        for chain in chains:
            if len(chain):
                skip = False
                break
        if skip:
            continue
        X.append([(0, 0, 0) for _ in range(4)])  # begin symbol is global symbol, update coordination afterwards
        S.append(VOCAB.symbol_to_idx(box))
        start = len(X)
        for chain in chains:
            for i in range(len(chain)):  # some chains do not participate
                residue = chain.get_residue(i)
                coord = residue.get_coord_map()
                x = []
                for atom in ['N', 'CA', 'C', 'O']:
                    if atom in coord:
                        x.append(coord[atom])
                    else:
                        coord[atom] = (0, 0, 0)
                        x.append((0, 0, 0))
                        corrupted_idx.append(len(X))
                        # print_log(f'Missing backbone atom coordination: {atom}', level='WARN')

                X.append(np.array(x))
                S.append(VOCAB.symbol_to_idx(residue.get_symbol()))
        X[start - 1] = np.mean(X[start:], axis=0)  # coordinate of global node
        chain_start_ends.append((start - 1, len(X)))

    # deal with corrupted coordinates
    for i in corrupted_idx:
        l, r = i - 1, i + 1
        if l > 0 and r < len(X):  # if at start / end, then leave it be
            X[i] = (X[l] + X[r]) / 2

    # set CDR pos for heavy chain
    offset = S.index(VOCAB.symbol_to_idx(VOCAB.BOH)) + 1
    L = ['0' for _ in X]

    for i in range(1, 4):
        begin, end = cplx.get_cdr_pos(f'H{i}')
        begin += offset
        end += offset
        for pos in range(begin, end + 1):
            L[pos] = str(i)

    res = {
        'X': torch.tensor(np.array(X), dtype=torch.float), # 3d coordination [n_node, 4, 3]
        'S': torch.tensor(S, dtype=torch.long),  # 1d sequence     [n_node]
        'L': ''.join(L)                          # cdr annotation, str of length n_node, 1 / 2 / 3 for cdr H1/H2/H3
    }
    return res


def main(args):

    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # get data
    data, raw_data = [], []
    for pdb, H, L in zip(args.pdb, args.heavy_chain, args.light_chain):
        protein = Protein.from_pdb(pdb)
        pdb_id = basename(splitext(pdb)[0])
        antigen_chains = [c for c in protein.get_chain_names() if c != H and c != L]
        cplx = AAComplex(pdb_id, protein.peptides, H, L, antigen_chains)
        data.append(to_tensor(cplx))
        raw_data.append((pdb_id, cplx))

    batch = EquiAACDataset.collate_fn(data)

    # generate
    with torch.no_grad():
        _, seqs, xs, _, _ = model.infer(batch, device)

    # save antibody
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    else:
        print(f'WARNING: {args.out_dir} exists! Files in it might be overwritten!')

    summary = open(os.path.join(args.out_dir, 'cdrs.txt'), 'w')
    summary.write(f'name\tCDR-{args.cdr_type}\n')

    for i in range(len(raw_data)):
        pdb_id, ori_cplx = raw_data[i]
        seq, x = seqs[i], xs[i]
        cplx = set_cdr(ori_cplx, seq, x, cdr=args.cdr_type)
        cplx.to_pdb(os.path.join(args.out_dir, pdb_id + '.pdb'))
        summary.write(f'{pdb_id}\t{seq}\n')
    summary.close()

    print(f'CDR sequences saved to {os.path.join(args.out_dir, "cdrs.txt")}')
    print(f'Results saved to {args.out_dir}')
    print('Done!')


def parse():
    parser = argparse.ArgumentParser(description='CDR design')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/ckpt/rabd_cdrh3_mean.ckpt',
                        help='Path to the checkpoint')
    parser.add_argument('--cdr_type', type=str, choices=['H3'], default='H3',
                        help='Type of cdr to design. Note that this should be consistent with training objective of the checkpoint.')

    parser.add_argument('--pdb', type=str, required=True, nargs='+', help='Path to the pdb of the docked antigen-antibody complex')
    parser.add_argument('--heavy_chain', type=str, required=True, nargs='+', help='Id of the heavy chain')
    parser.add_argument('--light_chain', type=str, required=True, nargs='+', help='Id of the light chain')
    parser.add_argument('--out_dir', type=str, default='./results', help='Path to save the generated antibodies')
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    return parser.parse_args()


if __name__ == '__main__':
    '''
    Example:
    1. Renumber pdb according to IMGT: python ./data/ImmunoPDB.py -i ./data/1ic7.pdb -o 1ic7.pdb -s imgt
        - for 1ic7.pdb, the heavy chain is H and the light chain is L
    2. Generate CDR-H3: python ./scripts/design.py --pdb 1ic7.pdb --heavy_chain H --light_chain L
        - the generated pdb will be saved to the folder ./results
    '''
    main(parse())