#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import time

from Bio.PDB import PDBIO

from data.pdb_utils import Peptide

FILE_DIR = os.path.split(__file__)[0]
TMEXEC = os.path.join(FILE_DIR, 'TMscore')
CACHE_DIR = os.path.join(FILE_DIR, '__tmcache__')
# create cache dir
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def tm_score(chain1: Peptide, chain2: Peptide):
    io, paths = PDBIO(), []
    for i, chain in enumerate([chain1, chain2]):
        chain_name = f'{time.time()}_{chain.get_id()}_{chain.seq[:20]}'  # for concurrent conflicts
        path = os.path.join(CACHE_DIR, f'{chain_name}.pdb')
        paths.append(path)
        chain = chain.to_bio()
        io.set_structure(chain)
        io.save(path)
    p = os.popen(f'{TMEXEC} {paths[0]} {paths[1]}')
    text = p.read()
    p.close()
    res = re.search(r'TM-score\s*= ([0-1]\.[0-9]+)', text)
    score = float(res.group(1))
    for path in paths:
        os.remove(path)
    return score