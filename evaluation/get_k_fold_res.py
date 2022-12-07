#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
from collections import defaultdict
import os
import re
import numpy as np


def parse():
    parser = argparse.ArgumentParser(description='Get K fold evaluation results')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory to k fold data')
    parser.add_argument('--cdr_type', type=int, choices=[1, 2, 3], required=True,
                        help='Type of cdr of the model')
    parser.add_argument('--model', type=str, required=True, help='Type of model (model name)')
    parser.add_argument('--mode', type=str, choices=['100', '111'], required=True,
                        help='Input mode, H/L/X, e.g. 111 for input H + L + X')
    parser.add_argument('--version', type=int, default=0, help='Version')
    return parser.parse_args()


def get_res_from_log(log_path):
    with open(log_path, 'r') as fin:
        lines = fin.readlines()[-10:]
    keys = ['PPL', 'RMSD', 'TMscore', 'AAR']
    patterns = [key + r': mean (\-?\d+\.\d+), std (\-?\d+\.\d+)' for key in keys]
    result = {}
    for i, key in enumerate(keys):
        p = patterns[i]
        for l in lines:
            match = re.search(p, l)
            if match is not None:
                mean, std = match.group(1), match.group(2)
                result[key] = (float(mean), float(std)) 
                break
    return result


def main(args):
    dirs = []
    for f in os.listdir(args.data_dir):
        k = re.match(r'fold_(\d+)', f)
        if k is None:
            continue
        dirs.append(os.path.join(args.data_dir, f))
    print(f'Number of fold detected: {len(dirs)}')
    sub_path = ['ckpt', f'{args.model}_CDR{args.cdr_type}_{args.mode}', f'version_{args.version}', 'eval_log.txt']
    all_res = defaultdict(list)
    for d in dirs:
        eval_f = d
        for p in sub_path:
            eval_f = os.path.join(eval_f, p)
        res_dict = get_res_from_log(eval_f)
        for key in res_dict:
            all_res[key].append(res_dict[key][0])
    for key in all_res:
        vals = all_res[key]
        print(f'{key}: mean {np.mean(vals)}, std {np.std(vals)}')


if __name__ == '__main__':
    main(parse())