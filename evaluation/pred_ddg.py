#!/usr/bin/python
# -*- coding:utf-8 -*-
from functools import partial
import os

import torch
from torch.multiprocessing import Pool
# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass

FILE_DIR = os.path.abspath(os.path.split(__file__)[0])
MODULE_DIR = os.path.join(FILE_DIR, 'ddg')
from .ddg.models.predictor import DDGPredictor
from .ddg.utils.misc import *
from .ddg.utils.data import *
from .ddg.utils.protein import *

CKPT = torch.load(os.path.join(MODULE_DIR, 'data', 'model.pt'))
MODEL = DDGPredictor(CKPT['config'].model)
MODEL.load_state_dict(CKPT['model'])
DEVICE = torch.device('cuda:0')
MODEL.to(DEVICE)
MODEL.eval()


# set the path to fixbb application of rosetta if you want to do ddG prediction with sidechain packing
ROSETTA_EXE = '/path/to/rosetta.binary.linux.release-315/main/source/bin/fixbb.static.linuxgccrelease'
RESFILE = f'{FILE_DIR}/resfile.txt'
CACHE_DIR=os.path.join(FILE_DIR, '__ddgcache__')
# create cache dir
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def pred_ddg_only(wt_pdb, mut_pdb):
    batch = load_wt_mut_pdb_pair(wt_pdb, mut_pdb)
    batch = recursive_to(batch, DEVICE)

    with torch.no_grad():
        pred = MODEL(batch['wt'], batch['mut']).item()
    return pred


def sidechain_opt_pred_ddg(wt_pdb, mut_pdb, cache_prefix=''):
    if not os.path.exists(ROSETTA_EXE):
        raise ValueError(f'Please provide the path to fixbb application of rosetta at ROSETTA_EXE ({__file__})')
    mut_pdb = os.path.abspath(mut_pdb)
    cmd = f'{ROSETTA_EXE} -in:file:s {mut_pdb} -in:file:fullatom -resfile {RESFILE} ' +\
            f'-nstruct 1 -out:path:all {CACHE_DIR} -out:prefix {cache_prefix} -overwrite -mute all'
    p = os.popen(cmd)
    p.read()
    p.close()
    filename = os.path.split(mut_pdb)[-1]
    tmp_pdb = os.path.join(CACHE_DIR, cache_prefix + filename[:-4] + '_0001' + '.pdb')
    pred =  pred_ddg_only(wt_pdb, tmp_pdb)
    os.remove(tmp_pdb)
    return pred


def pred_ddg(wt_pdb, mut_pdb, opt_sidechain=0, n_proc=1):
    if opt_sidechain > 0:
        if opt_sidechain == 1 or n_proc == 1:
            scores = [sidechain_opt_pred_ddg(wt_pdb, mut_pdb, str(i)) for i in range(opt_sidechain)]
        else:
            raise NotImplementedError('Parallel prediction with sidechain packing is not implemented')
            pool = Pool(n_proc)
            scores = pool.map(partial(sidechain_opt_pred_ddg, wt_pdb, mut_pdb), [str(i) for i in range(opt_sidechain)])
            pool.close()
        return sum(scores) * 1.0 / len(scores)
    else:
        return pred_ddg_only(wt_pdb, mut_pdb)
