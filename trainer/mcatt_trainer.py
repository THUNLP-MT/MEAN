#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch

from .abs_trainer import Trainer


class MCAttTrainer(Trainer):

    # define optimizer
    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    # scheduler example: linear. Return None if no scheduler is needed.
    def get_scheduler(self, optimizer):
        lam = lambda epoch: self.config.anneal_base ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
        return {
            'scheduler': scheduler,
            'frequency': 'epoch' # or batch
        }

    # train step, note that batch should be dict/list/tuple or other objects with .to(device) attribute
    def train_step(self, batch, batch_idx):
        return self.share_forward(batch, batch_idx, 'train')

    # validation step
    def valid_step(self, batch, batch_idx):
        return self.share_forward(batch, batch_idx, 'validation', val=True)

    def share_forward(self, batch, batch_idx, _type, val=False):
        loss, snll, closs = self.model(
            batch['X'], batch['S'], batch['L'], batch['offsets']
        )
        ppl = snll.exp().item()
        self.log(f'Loss/{_type}', loss, batch_idx, val=val)
        self.log(f'SNLL/{_type}', snll, batch_idx, val=val)
        self.log(f'Closs/{_type}', closs, batch_idx, val=val)
        self.log(f'PPL/{_type}', ppl, batch_idx, val=val)
        return loss