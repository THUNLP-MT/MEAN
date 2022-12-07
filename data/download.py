#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import requests
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map

from .pdb_utils import Protein, AAComplex
from utils.io import read_csv
from utils.network import url_get


def fetch_from_pdb(identifier, tries=5):
    # example identifier: 1FBI

    identifier = identifier.upper()
    url = 'https://data.rcsb.org/rest/v1/core/entry/' + identifier

    res = url_get(url, tries)
    if res is None:
        return None

    url = f'https://files.rcsb.org/download/{identifier}.pdb'

    text = url_get(url, tries)
    data = res.json()
    data['pdb'] = text.text
    return data


'''SAbDab Summary reader'''

def fetch_from_sabdab(identifier, tries=5):
    # example identifier: 1mhp
    url = 'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/structureviewer/?pdb=' + identifier.lower()
    
    res = url_get(url, tries)
    if res is None:
        return None

    try:
        item = {'pdb': identifier}
        num_fv = re.search(r'This PDB has (\d+) Fv\(s\).', res.text).group(1)
        assert num_fv.isdigit(), f'{num_fv} is not a integer'
        num_fv = int(num_fv)
        key_map = {
            'heavy_chain': 'Heavy chain',
            'light_chain': 'Light chain',
            'antigen_chains': 'Antigen chains',
            'antigen_seqs': 'Antigen sequence'
        }
        filled_cnts = [0 for _ in range(num_fv)]
        results = {key: [] for key in key_map}
        for key in key_map:
            pattern = r'<b>' + key_map[key] + r'</b>\s*</td>\s*<td width="50%">(.*?)</td>'
            hits = re.findall(pattern, res.text, re.I | re.M)
            for fv in range(num_fv):
                if fv < len(hits) and hits[fv] != '':
                    filled_cnts[fv] += 1
            results[key] = hits
        best_fv = max([i for i in range(num_fv)], key=lambda i: filled_cnts[i])
        for key in key_map:
            if best_fv < len(results[key]):
                item[key] = results[key][best_fv]
            else:
                item[key] = ''
        for key, splitter in zip(['antigen_chains', 'antigen_seqs'], [',', '/']):
            if len(item[key]):
                item[key] = item[key].split(splitter)
            else:
                item[key] = []

        # for chain in ['heavy_chain', 'light_chain']:
        #     if item[chain] == '':
        #         continue
        #     seq_raw = re.findall(r'<p class="alignmentheader">' + key_map[chain] + r'</p>\s*<div style="width=100%; overflow-x: scroll;">\s*<table class="table table-alignment">\s*<tr>\s*(.*?)\s*</tr>\s*</table>', res.text, re.I | re.M | re.S)[0]
        #     seq, cdr_annotate = '', ''
        #     seq_raw = [line.strip() for line in seq_raw.split('\n')]
        #     seperate_pos = seq_raw.index('</tr>')
        #     cdr_start, cdr_cnt = -1, 0
        #     for data in seq_raw[:seperate_pos]:
        #         if 'background-color' in data:
        #             if cdr_start == -1:
        #                 cdr_cnt += 1
        #                 cdr_start = len(cdr_annotate)
        #             cdr_annotate += str(cdr_cnt)
        #         else:
        #             cdr_start = -1
        #             cdr_annotate += '0'
        #     assert cdr_cnt == 3, f'Number of CDR is {cdr_cnt}, not 3'
        #     for data in seq_raw[seperate_pos + 2:]:
        #         residue_sym = re.findall(r'<td>(.*?)</td>', data)[0]
        #         seq += residue_sym
        #     for i in range(1, 4):
        #         i = str(i)
        #         start, end = cdr_annotate.index(i), cdr_annotate.rindex(i)
        #         cdr = f'cdr{chain[0]}{i}'
        #         item[cdr + '_pos'] = (start, end)
        #         item[cdr + '_seq'] = seq[start:end + 1]
        #     item[f'{chain}_seq'] = seq

        return item
    except AssertionError as e:
        print(f'fetch {identifier} failed because of assertion error {e}, skip')
        return None
    except IndexError as e:
        print(f'fetch {identifier} failed because of index error {e}, skip')
        return None


def read_sabdab(fpath, n_cpu):
    heads, entries = read_csv(fpath, sep='\t')
    head2idx = { head: i for i, head in enumerate(heads) }
    items, pdb2idx = [], {}
    head_mapping = {
        'heavy_chain': head2idx['Hchain'],
        'light_chain': head2idx['Lchain'],
        'antigen_chains': head2idx['antigen_chain']
    }
    for entry in entries:
        pdb = entry[head2idx['pdb']]
        if pdb in pdb2idx:  # reserve the one with better resolution
            continue
        item = { 'pdb': pdb }
        for head in head_mapping:
            item[head] = entry[head_mapping[head]]
            if item[head] == 'nan':
                item[head] = ''
        if len(item['antigen_chains']):
            item['antigen_chains'] = item['antigen_chains'].split(' | ')
        else:
            item['antigen_chains'] = []
        for chain_name in item['antigen_chains']:
            if chain_name == item['heavy_chain'] or chain_name == item['light_chain']:  # the antigen is the antibody (actually no antigen in the pdb)
                item['antigen_chains'] = []
                break
        if item['heavy_chain'].upper() == item['light_chain'].upper():
            item['light_chain'] = ''  # actually no light chain
        if pdb in pdb2idx:
            items[pdb2idx[pdb]] = item
        else:
            pdb2idx[pdb] = len(items)
            items.append(item)
    # example of an item:
    # {
    #   'pdb': xxxx,
    #   'heavy_chain': A,
    #   'light_chain': B,
    #   'antigen_chains': [N, L]
    # }
    return items


'''RABD Summary reader'''
def read_rabd(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    items = [json.loads(s.strip('\n')) for s in lines]
    keys = ['pdb', 'heavy_chain', 'light_chain', 'antigen_chains']
    for item in items:
        for key in keys:
            assert key in item, f'{item} do not have {key}'
    return items


'''Sceptre Summary reader'''

def read_sceptre(fpath):
    head, entries = read_csv(fpath)
    head_mapping = {
        'pdb': 'pdb_id',
        'heavy_chain': 'ab_c1_pdb_chain',
        'light_chain': 'ab_c2_pdb_chain',
        'antigen_chains': 'ant_pdb_chain',
        'heavy_chain_seq': 'chain1_full_seq',
        'light_chain_seq': 'chain2_full_seq'
    }
    for name, chain_id in zip(['h', 'l'], [1, 2]):
        for i in range(1, 4):
            cdr = f'cdr{name}{chain_id}'
            head_mapping[cdr + '_pos'] = [
                f'chain{chain_id}_cdr{i}_start_calculated',
                f'chain{chain_id}_cdr{i}_end_calculated',
            ]
            head_mapping[cdr + '_seq'] = f'chain{chain_id}_cdr{i}_seq_calculated'

    head2idx = {}
    for i, h in enumerate(head):
        head2idx[h] = i

    items = []
    for entry in entries:
        item = {}
        for key in head_mapping:
            origin_key = head_mapping[key]
            if isinstance(origin_key, list):
                item[key] = [entry[head2idx[k]] for k in origin_key]
            else:
                idx = head2idx[origin_key]
                item[key] = entry[idx]
        items.append(item)
    return items
    

def download_one_item(item):
    pdb_id = item['pdb']
    pdb_data = fetch_from_pdb(pdb_id)
    if pdb_data is None:
        print(f'{pdb_id} invalid')
        item = None
    else:
        item['pdb_data'] = pdb_data['pdb']
    return item


def download_one_item_local(pdb_dir, item):
    pdb_id = item['pdb']
    for pdb_id in [pdb_id.lower(), pdb_id.upper()]:
        fname = os.path.join(pdb_dir, pdb_id + '.pdb')
        if not os.path.exists(fname):
            continue
        with open(fname, 'r') as fin:
            item['pdb_data'] = fin.read()
        return item
    print(f'{pdb_id} not found in {pdb_dir}, try fetching from remote server')
    from_remote = fetch_from_pdb(pdb_id)
    if from_remote is not None:
        print('fetched')
        item['pdb_data'] = from_remote['pdb']
        item['pre_numbered'] = False
        return item
    return None


def post_process(item):
    # renumbering pdb file and revise cdrs
    pdb_data_path, numbering, pdb = item['pdb_data_path'], item['numbering'], item['pdb']
    if numbering == 'none':
        pass
    elif numbering == 'imgt':
        if not item['pre_numbered']:
            cmd = f'python ImmunoPDB.py -i {pdb_data_path} -o {pdb_data_path} -s imgt'
            exit_code = os.system(cmd)
            if exit_code != 0:
                print(f'renumbering failed for {pdb}. scheme {numbering}')
                return None
        try:
            protein = Protein.from_pdb(pdb_data_path)
            cplx = AAComplex(pdb, protein.peptides, item['heavy_chain'],
                             item['light_chain'], item['antigen_chains'],
                             numbering=numbering, skip_cal_interface=True)
        except (AssertionError, ValueError) as e:
            print(f'parsing pdb failed for {pdb}. reason: {e}')
            return None
        except Exception as e:
            print(f'parsing pdb failed for {pdb} unexpectedly. reason: {e}')
            return None
        item['heavy_chain_seq'] = cplx.get_heavy_chain().get_seq()
        item['light_chain_seq'] = cplx.get_light_chain().get_seq()
        item['antigen_seqs'] = [ chain.get_seq() for chain in cplx.get_antigen_chains() ]
        for c in ['H', 'L']:
            for i in range(1, 4):
                cdr_name = f'{c}{i}'.lower()
                cdr_pos, cdr = cplx.get_cdr_pos(cdr_name), cplx.get_cdr(cdr_name)
                item[f'cdr{cdr_name}_pos'] = cdr_pos
                item[f'cdr{cdr_name}_seq'] = cdr.get_seq()
    else:
        raise NotImplementedError(f'Numbering scheme {numbering} not supported')
    return item


def download(items, out_path, ncpu=8, pdb_dir=None, numbering='imgt', pre_numbered=False):
    if pdb_dir is None:
        map_func = download_one_item
    else:
        map_func = partial(download_one_item_local, pdb_dir)
    print('downloading raw files')
    valid_entries = thread_map(map_func, items, max_workers=ncpu)
    valid_entries = [item for item in valid_entries if item is not None]
    print(f'number of downloaded entries: {len(valid_entries)}')
    pdb_out_dir = os.path.join(os.path.split(out_path)[0], 'pdb')
    if os.path.exists(pdb_out_dir):
        print(f'WARNING: pdb file out directory {pdb_out_dir} exists!')
    else:
        os.makedirs(pdb_out_dir)
    print(f'writing PDB files to {pdb_out_dir}')
    for item in tqdm(valid_entries):
        pdb_fout = os.path.join(pdb_out_dir, item['pdb'] + '.pdb')
        with open(pdb_fout, 'w') as pfout:
            pfout.write(item['pdb_data'])
        item.pop('pdb_data')
        item['pdb_data_path'] = os.path.abspath(pdb_fout)
        item['numbering'] = numbering
        if 'pre_numbered' not in item:
            item['pre_numbered'] = pre_numbered
    print('post processing')
    valid_entries = process_map(post_process, valid_entries, max_workers=ncpu, chunksize=1)
    valid_entries = [item for item in valid_entries if item is not None]
    print(f'number of valid entries: {len(valid_entries)}')
    fout = open(out_path, 'w')
    for item in valid_entries:
        item_str = json.dumps(item)
        fout.write(f'{item_str}\n')
    fout.close()
    return valid_entries


def statistics(items, n_eg=5):  # show n_eg instances for each type of data
    keys = ['heavy_chain', 'light_chain', 'antigen_chains']
    cnts, example = defaultdict(int), {}
    for item in items:
        res = ''
        for key in keys:
            res += '1' if len(item[key]) else '0'
        cnts[res] += 1
        if res not in example:
            example[res] = []
        if len(example[res]) < n_eg:
            example[res].append(item['pdb'])
    sorted_desc_keys = sorted(list(cnts.keys()))
    for desc_key in sorted_desc_keys:
        desc = 'Only has '
        for key, val in zip(keys, desc_key):
            if val == '1':
                desc += key + ', '
        desc += str(cnts[desc_key])
        print(f'{desc}, examples: {example[desc_key]}')


def parse():
    parser = ArgumentParser(description='download full pdb data')
    parser.add_argument('--summary', type=str, required=True, help='Path to summary file')
    parser.add_argument('--fout', type=str, required=True, help='Path to output json file')
    parser.add_argument('--type', type=str, choices=['sceptre', 'sabdab', 'rabd'], default='sceptre',
                        help='Type of the dataset')
    parser.add_argument('--pdb_dir', type=str, default=None, help='Path to local folder of PDB files')
    parser.add_argument('--numbering', type=str, default='imgt', choices=['imgt', 'none'],
                        help='Renumbering scheme')
    parser.add_argument('--pre_numbered', action='store_true', help='The files in pdb_dir is already renumbered')
    parser.add_argument('--n_cpu', type=int, default=8, help='Number of cpu to use')
    return parser.parse_args()


def main(args):
    fpath, out_path = args.summary, args.fout
    print(f'download {args.type} from summary file {fpath}')
    if args.pdb_dir is not None:
        print(f'using local PDB files: {args.pdb_dir}')
        if args.pre_numbered:
            print(f'PDB file already renumbered with scheme {args.numbering}')
    if args.type == 'sceptre':
        items = read_sceptre(fpath)
    elif args.type == 'sabdab':
        items = read_sabdab(fpath, args.n_cpu)
    elif args.type == 'rabd':
        items = read_rabd(fpath)
    items = download(items, out_path, args.n_cpu, args.pdb_dir, args.numbering, args.pre_numbered)
    statistics(items)


if __name__ == '__main__':
    main(parse())