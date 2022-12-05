#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch

from .abs_trainer import Trainer


class Seq2SeqTrainer(Trainer):

    # define optimizer
    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    # scheduler example: linear. Return None if no scheduler is needed.
    def get_scheduler(self, optimizer):
        return None

    # train step, note that batch should be dict/list/tuple or other objects with .to(device) attribute
    def train_step(self, batch, batch_idx):
        snll = self.model(batch['S'], batch['L'], batch['mask'])
        ppl = snll.exp().item()
        self.log('Loss/train',snll, batch_idx)
        self.log('PPL/train', ppl, batch_idx)
        return snll

    # validation step
    def valid_step(self, batch, batch_idx):
        snll = self.model(batch['S'], batch['L'], batch['mask'])
        ppl = snll.exp().item()
        self.log('Loss/train',snll, batch_idx)
        self.log('PPL/train', ppl, batch_idx)
        return snll