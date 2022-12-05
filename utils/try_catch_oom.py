#!/usr/bin/python
# -*- coding:utf-8 -*-
from .logger import print_log

def try_catch_oom(exec_func, *args, **kwargs):
    try:
        res = exec_func(*args, **kwargs)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print_log(f'OOM caught executing {exec_func}', level='ERROR')
            return None
        else:
            raise e
    return res
