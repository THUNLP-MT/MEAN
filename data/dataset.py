#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import deepcopy
import functools
import os
import json
import pickle
import argparse
from typing import List

import numpy as np
import torch

from utils.logger import print_log

########## import your packages below ##########
from tqdm import tqdm
import torch

from data.pdb_utils import AAComplex, Protein, VOCAB



# use this class to splice the dataset and maintain only one part of it in RAM
class EquiAACDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, save_dir=None, num_entry_per_file=-1, random=False, ctx_cutoff=8.0, interface_cutoff=12.0):
        '''
        file_path: path to the dataset
        save_dir: directory to save the processed data
        num_entry_per_file: number of entries in a single file. -1 to save all data into one file 
                            (In-memory dataset)
        '''
        super().__init__()
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        self.data: List[AAComplex] = []  # list of ABComplex

        # try loading preprocessed files
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = metainfo['file_names']
                self.file_num_entries = metainfo['file_num_entries']
        except Exception as e:
            print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True

        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '111'  # H/L/Antigen, 1 for include, 0 for exclude
        self.ctx_cutoff = ctx_cutoff
        self.interface_cutoff = interface_cutoff

    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx
     
    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        for line in tqdm(lines):
            item = json.loads(line)
            try:
                protein = Protein.from_pdb(item['pdb_data_path'])
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            pdb_id, peptides = item['pdb'], protein.peptides
            self.data.append(AAComplex(pdb_id, peptides, item['heavy_chain'], item['light_chain'], item['antigen_chains']))
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    def __getitem__(self, idx):
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        item, res = self.data[idx], {}
        # each item is an instance of ABComplex. res has following entries
        # X: [seq_len, 5, 3], coordinates of N, CA, C, O, CB, center of side chain. Missing data are set to the average of two ends
        # S: [seq_len], indices of each residue
        # L: string of cdr labels, 0 for non-cdr residues, 1 for cdr1, 2 for cdr2, 3 for cdr3 

        hc, lc = item.get_heavy_chain(), item.get_light_chain()
        antigen_chains = item.get_antigen_chains(interface_only=True, cdr=None)

        # prepare input
        chain_lists, begins = [], []
        if self.mode[0] == '1': # the cdrh3 start pos in a batch should be near (for RefineGNN)
            chain_lists.append([hc])
            begins.append(VOCAB.BOH)
        if self.mode[1] == '1':
            chain_lists.append([lc])
            begins.append(VOCAB.BOL)
        if self.mode[2] == '1':
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
            X.append([(0, 0, 0) for _ in range(5)])  # begin symbol is global symbol, update coordination afterwards
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
            begin, end = item.get_cdr_pos(f'H{i}')
            begin += offset
            end += offset
            for pos in range(begin, end + 1):
                L[pos] = str(i)

        res = {
            'X': torch.tensor(np.array(X), dtype=torch.float), # 3d coordination [n_node, 5, 3]
            'S': torch.tensor(S, dtype=torch.long),  # 1d sequence     [n_node]
            'L': ''.join(L)                          # cdr annotation, str of length n_node, 1 / 2 / 3 for cdr H1/H2/H3
        }
        return res

    def __len__(self):
        return self.num_entry

    @classmethod
    def collate_fn(cls, batch):
        Xs, Ss, Ls = [], [], []
        offsets = [0]
        for i, data in enumerate(batch):
            Xs.append(data['X'])
            Ss.append(data['S'])
            Ls.append(data['L'])
            offsets.append(offsets[-1] + len(Ss[i]))

        return {
            'X': torch.cat(Xs, dim=0),  # [n_all_node, 5, 3]
            'S': torch.cat(Ss, dim=0),  # [n_all_node]
            'L': Ls,
            'offsets': torch.tensor(offsets, dtype=torch.long)
        }


class ITAWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, n_samples, _cmp=lambda score1, score2: score1 - score2):
        super().__init__()
        self.dataset = deepcopy(dataset)
        self.dataset._check_load_part = lambda idx: idx
        self.candidates = [[(self.dataset.data[i], 0)] for i in self.dataset.idx_mapping]
        self.n_samples = n_samples
        self.cmp = _cmp

    def _cmp_wrapper(self, a, b):  # tuple of (cplx, score)
        return self.cmp(a[1], b[1])

    def update_candidates(self, i, candidates): # tuple of (cplx, score)
        all_cand = candidates + self.candidates[i]
        all_cand.sort(key=functools.cmp_to_key(self._cmp_wrapper))
        self.candidates[i] = all_cand[:self.n_samples]

    def finish_update(self):  # update all candidates to dataset
        data, idx_mapping = [], []
        for candidates in self.candidates:
            for cand, score in candidates:
                idx_mapping.append(len(data))
                data.append(cand)
        self.dataset.data = data
        self.dataset.idx_mapping = idx_mapping

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        return self.dataset.collate_fn(batch)


def parse():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save processed data')
    return parser.parse_args()
 

if __name__ == '__main__':
    args = parse()
    dataset = EquiAACDataset(args.dataset, args.save_dir, num_entry_per_file=-1)
    print(len(dataset))