#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
from collections import defaultdict
from functools import partial
from tqdm.contrib.concurrent import process_map

import numpy as np

from data import AACDataset, AAComplex


def aggr_results(results):
    final_res = {}
    all_keys = {}
    for item in results:
        all_keys.update(item)
    for key in all_keys:
        vals = [(res[key] if key in res else 0) for res in results]
        if isinstance(vals[0], dict):
            val = aggr_results(vals)
        else:
            val = np.mean(vals)
        final_res[key] = val
    return final_res


def print_dict(d, i=0):
    for key in d:
        val = d[key]
        if isinstance(val, dict):
            print('\t' * i + f'{key}: ')
            print_dict(val, i + 1)
        else:
            print('\t' * i + f'{key}: {val}')


def get_all_xs(cplx: AAComplex):
    xs, mask_id, offset_mapping = [], [], {}
    for chain in cplx.get_chain_names():
        c = cplx.get_chain(chain)
        offset_mapping[chain] = [len(xs), len(xs) + len(c)] # [begin, end)
        for i in range(len(c)):
            try:
                ca_pos = c.get_cb_pos(i)
            except KeyError:
                ca_pos = [0, 0, 0]
                mask_id.append(len(xs))
            xs.append(ca_pos)
    xs = np.array(xs)
    return xs, mask_id, offset_mapping


# count ratio of different node types in a KNN graph
def k_neighbors_antigen_cnt(cplx: AAComplex, k_neighbors=9):
    res = {}
    xs, mask_id, offset_mapping = get_all_xs(cplx)
    dist = np.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
    for i in mask_id:
        dist[i, :] = 1e10
        dist[:, i] = 1e10

    h_start, h_end = offset_mapping[cplx.heavy_chain]
    l_start, l_end = offset_mapping[cplx.light_chain] if cplx.light_chain in offset_mapping else (-1, -1)
    for cdr in ['H1', 'H2', 'H3']:
        cdr_pos = cplx.get_cdr_pos(cdr)
        offset, _ = offset_mapping[cplx.heavy_chain]
        start, end = offset + cdr_pos[0], offset + cdr_pos[1]
        node_cnt, h_cnt, l_cnt = 0, 0, 0
        for i in range(start, end + 1):
            if i in mask_id:
                continue
            cdr_dist = dist[i]
            k = k_neighbors + 1
            ind = np.argpartition(cdr_dist, k)[:k]  # the first must be itself
            ind = ind[np.argsort(cdr_dist[ind])]
            # assert ind[0] == i, f'{i} {ind}, {xs[i]} {xs[ind[0]]}, {cplx}'
            # ind = ind[1:]
            ind = ind.tolist()
            ind.remove(i)
            for j in ind:
                if j >= h_start and j < h_end:
                    h_cnt += 1
                elif j >= l_start and j < l_end:
                    l_cnt += 1
                node_cnt += 1
        res[cdr] = {
            'heavy_ratio': h_cnt * 1.0 / node_cnt,
            'light_ratio': l_cnt * 1.0 / node_cnt,
            'antigen_ratio': 1 - (h_cnt + l_cnt) * 1.0 / node_cnt
        }
    return res


def judge_type(i, cplx, offset_mapping):
    h_start, h_end = offset_mapping[cplx.heavy_chain]
    l_start, l_end = offset_mapping[cplx.light_chain] if cplx.light_chain in offset_mapping else (-1, -1)
    if i >= h_start and i < h_end:
        chain, in_cdr = 'Heavy', None
        for cdr in ['H1', 'H2', 'H3']:
            cdr_pos = cplx.get_cdr_pos(cdr)
            start, end = h_start + cdr_pos[0], h_start + cdr_pos[1]
            if i >= start and i <= end:
                in_cdr = cdr
                break
        cdr = in_cdr
    elif i >= l_start and i < l_end:
        chain, in_cdr = 'Light', None
        for cdr in ['L1', 'L2', 'L3']:
            cdr_pos = cplx.get_cdr_pos(cdr)
            start, end = h_start + cdr_pos[0], h_end + cdr_pos[1]
            if i >= start and i <= end:
                in_cdr = cdr
                break
        cdr = in_cdr
    else:
        chain, cdr = 'Antigen', None
    return (chain, cdr)


# count ratio of different node types in interface pairs
def interface_pairs_type(cplx: AAComplex):
    cutoff = AAComplex.threshold
    xs, mask_id, offset_mapping = get_all_xs(cplx)
    dist = np.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
    for i in mask_id:
        dist[i, :] = 1e10
        dist[:, i] = 1e10

    cnts = defaultdict(int)
    _is, _js = (dist < cutoff).nonzero()
    for i, j in zip(_is, _js):
        if i >= j:  # down triangle
            continue
        i_chain, i_cdr = judge_type(i, cplx, offset_mapping)
        j_chain, j_cdr = judge_type(j, cplx, offset_mapping)
        if i_chain == j_chain:  # not interface
            continue
        i_name = i_chain if i_cdr is None else i_cdr
        j_name = j_chain if j_cdr is None else j_cdr
        if i_name < j_name:
            i_name, j_name = j_name, i_name
        cnts[f'{i_name}-{j_name}'] += 1
    return cnts


# count ratio of different node types in interface pairs based on nearest atoms
def interface_pairs_type_atom_dist(cplx: AAComplex):
    cutoff = AAComplex.threshold
    residues, offset_mapping = [], {}
    for chain in cplx.get_chain_names():
        c = cplx.get_chain(chain)
        offset_mapping[chain] = [len(residues), len(residues) + len(c)] # [begin, end)
        for i in range(len(c)):
            residue = c.get_residue(i)
            residues.append(residue)

    cnts = defaultdict(int)
    for i in range(len(residues)):
        for j in range(i + 1, len(residues)):
            i_chain, i_cdr = judge_type(i, cplx, offset_mapping)
            j_chain, j_cdr = judge_type(j, cplx, offset_mapping)
            if i_chain == j_chain:  # not interface
                continue
            dist = residues[i].dist_to(residues[j])
            if dist is None or dist >= cutoff:
                continue
            i_name = i_chain if i_cdr is None else i_cdr
            j_name = j_chain if j_cdr is None else j_cdr
            if i_name < j_name:
                i_name, j_name = j_name, i_name
            cnts[f'{i_name}-{j_name}'] += 1
    return cnts


def do_stats(func, all_cplx, num_workers):
    res = process_map(func, all_cplx, max_workers=num_workers, chunksize=10)
    res = aggr_results(res)
    print_dict(res)


def parse():
    parser = argparse.ArgumentParser(description='statistics of antibody-antigen dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of cpu to use')
    return parser.parse_args()


def main(args):
    dataset = AACDataset(args.dataset)
    print(f'loaded number of complexes: {len(dataset)}')
    all_cplx = [cplx for cplx in dataset.data if cplx.heavy_chain.upper() != cplx.light_chain.upper()]
    print(f'valid number of complexes: {len(all_cplx)}')
    # k neighbors stat
    print('Node type statistics for K neighbors graph based on CA position')
    # do_stats(partial(k_neighbors_antigen_cnt, k_neighbors=9), all_cplx, args.num_workers)

    # interface pair type
    print('Interface pair (based on CA coordination) type statistics')
    do_stats(interface_pairs_type, all_cplx, args.num_workers)
    
    print('Interface pair (based on all atom coordination) type statistics')
    # do_stats(interface_pairs_type_atom_dist, all_cplx, args.num_workers)

if __name__ == '__main__':
    main(parse())