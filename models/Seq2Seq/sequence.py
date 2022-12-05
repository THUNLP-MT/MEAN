'''
This file is copied from RefineGNN and then modified according to our training framework
'''
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from data.pdb_utils import VOCAB

ReturnType = namedtuple('ReturnType',('nll','ppl','X','X_cdr'), defaults=(None, None, None, None))


class SeqModel(nn.Module):

    def __init__(self, args):
        super(SeqModel, self).__init__()
        self.hidden_size = args.hidden_size
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)

        self.lstm = nn.LSTM(
                args.hidden_size, args.hidden_size, 
                batch_first=True, num_layers=args.depth,
                dropout=args.dropout
        )
        self.W_out = nn.Linear(args.hidden_size, args.vocab_size, bias=True)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, S, mask):
        h_S = self.W_s(S)
        h_S_shift = F.pad(h_S[:,0:-1], (0,0,1,0), 'constant', 0)
        h_V, _ = self.lstm(h_S_shift)
        logits = self.W_out(h_V)
        loss = self.ce_loss(logits.view(-1, logits.size(-1)), S.view(-1))
        loss = torch.sum(loss * mask.view(-1)) / mask.sum()
        return loss

    def log_prob(self, S, mask):
        return ReturnType(nll=self(S, mask))

    def generate(self, B, N):
        h = torch.zeros(self.lstm.num_layers, B, self.hidden_size).cuda()
        c = torch.zeros(self.lstm.num_layers, B, self.hidden_size).cuda()
        S = torch.zeros(B, N + 1).long().cuda()
        for t in range(N):
            h_S = self.W_s(S[:, t:t+1])
            h_V, (h, c) = self.lstm(h_S, (h, c))
            logits = self.W_out(h_V)
            prob = F.softmax(logits, dim=-1).squeeze(1)
            S[:, t+1] = torch.multinomial(prob, num_samples=1).squeeze(-1)
        
        S = S[:, 1:].tolist()
        S = [''.join([VOCAB.idx_to_symbol(S[i][j]) for j in range(N)]) for i in range(B)]
        return S


class Seq2Seq(nn.Module):

    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.hidden_size = args.hidden_size
        self.cdr_type = args.cdr_type
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)
        self.W_a = nn.Embedding(args.vocab_size, args.hidden_size)

        self.encoder = nn.LSTM(
                args.hidden_size, args.hidden_size, 
                batch_first=True, num_layers=args.depth,
                dropout=args.dropout
        )
        self.decoder = nn.LSTM(
                args.hidden_size, args.hidden_size, 
                batch_first=True, num_layers=args.depth,
                dropout=args.dropout
        )
        self.W_out = nn.Linear(args.hidden_size * 2, args.vocab_size, bias=True)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def RL_parameters(self):
        return self.parameters()

    def encode(self, aS, amask):
        h_S = self.W_a(aS)
        h_V, _ = self.encoder(h_S)  # [B, M, H]
        return h_V * amask.unsqueeze(-1)

    def prepare_input(self, true_S, true_cdr, mask):
        cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in true_cdr]
        cdr_S, max_len, context_S = [], 0, true_S.clone()
        for i, (l, r) in enumerate(cdr_range):
            cdr_S.append(true_S[i, l:r + 1])
            context_S[i, l:r + 1] = VOCAB.get_unk_idx()
            max_len = max(max_len, len(cdr_S[-1]))
        cdr_mask = torch.zeros(len(cdr_S), max_len, device=true_S.device)
        for i, cdr in enumerate(cdr_S):
            cdr_mask[i, :len(cdr)] = 1
        cdr_S = pad_sequence(cdr_S, batch_first=True, padding_value=VOCAB.get_pad_idx())
        return cdr_S, cdr_mask, (context_S, mask, cdr_range)

    def forward(self, true_S, true_cdr, mask):
        S, smask, context = self.prepare_input(true_S, true_cdr, mask)
        return self.inner_forward(S, smask, context)

    def inner_forward(self, S, mask, context):
        h_S = self.W_s(S)
        h_S_shift = F.pad(h_S[:,0:-1], (0,0,1,0), 'constant', 0)
        h_V, _ = self.decoder(h_S_shift)  # [B, N, H]

        # attention
        aS, amask, _ = context
        h_A = self.encode(aS, amask)  # [B, M, H]
        att = torch.bmm(h_V, h_A.transpose(1, 2))  # [B, N, M]
        att_mask = mask.unsqueeze(2) * amask.unsqueeze(1)  # [B, N, 1] * [B, 1, M]
        att = att - (1 - att_mask) * 1e6  # attention mask
        att = F.softmax(att, dim=-1)  # [B, N, M]
        h_att = torch.bmm(att, h_A)
        h_out = torch.cat([h_V, h_att], dim=-1)

        logits = self.W_out(h_out)
        loss = self.ce_loss(logits.view(-1, logits.size(-1)), S.view(-1))
        loss = torch.sum(loss * mask.view(-1)) / mask.sum()
        return loss

    def log_prob(self, S, mask, context):
        B, N = S.size(0), S.size(1)
        h_S = self.W_s(S)
        h_S_shift = F.pad(h_S[:,0:-1], (0,0,1,0), 'constant', 0)
        h_V, _ = self.decoder(h_S_shift)  # [B, N, H]

        # attention
        aS, amask, _ = context
        h_A = self.encode(aS, amask)  # [B, M, H]
        att = torch.bmm(h_V, h_A.transpose(1, 2))  # [B, N, M]
        att_mask = mask.unsqueeze(2) * amask.unsqueeze(1)  # [B, N, 1] * [B, 1, M]
        att = att - (1 - att_mask) * 1e6  # attention mask
        att = F.softmax(att, dim=-1)  # [B, N, M]
        h_att = torch.bmm(att, h_A)
        h_out = torch.cat([h_V, h_att], dim=-1)

        logits = self.W_out(h_out)
        loss = self.ce_loss(logits.view(-1, logits.size(-1)), S.view(-1))
        loss = loss.view(B, N)
        ppl = torch.sum(loss * mask, dim=-1) / mask.sum(dim=-1)
        nll = torch.sum(loss * mask) / mask.sum()

        return ReturnType(nll=nll, ppl=ppl)

    def generate(self, B, N, context, return_ppl=False):
        aS, amask, _ = context
        h_A = self.encode(aS, amask)  # [B, M, H]

        h = torch.zeros(self.decoder.num_layers, B, self.hidden_size).cuda()
        c = torch.zeros(self.decoder.num_layers, B, self.hidden_size).cuda()
        S = torch.zeros(B, N + 1).long().cuda()
        sloss = 0.

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        special_mask = special_mask.repeat(B, 1).bool()  # [B, vocab_size]

        for t in range(N):
            h_S = self.W_s(S[:, t:t+1])
            h_V, (h, c) = self.decoder(h_S, (h, c))

            att = torch.bmm(h_V, h_A.transpose(1, 2))  # [B, 1, M]
            att_mask = amask.unsqueeze(1)  # [B, 1, M]
            att = att - (1 - att_mask) * 1e6  # attention mask
            att = F.softmax(att, dim=-1)  # [B, 1, M]
            h_att = torch.bmm(att, h_A)   # [B, 1, H]
            h_out = torch.cat([h_V, h_att], dim=-1)

            logits = self.W_out(h_out).squeeze(1)
            logits = logits.masked_fill(special_mask, float('-inf'))  # mask special tokens
            prob = F.softmax(logits, dim=-1)
            S[:, t+1] = torch.multinomial(prob, num_samples=1).squeeze(-1)
            sloss = sloss + self.ce_loss(logits, S[:, t+1])
        
        S = S[:, 1:].tolist()
        S = [''.join([VOCAB.idx_to_symbol(S[i][j]) for j in range(N)]) for i in range(B)]
        ppl = torch.exp(sloss / N)
        return (S, ppl) if return_ppl else S

    # unified api: return ppl (list), seq (list), x (numpy), true_x (numpy) and whether aligned
    def infer(self, batch, device):
        true_S = batch['S'].to(device)
        true_L = batch['L']
        mask = batch['mask'].to(device)
        cdr_S, cdr_mask, context = self.prepare_input(true_S, true_L, mask)
        B, N = cdr_S.shape
        seqs, ppls = self.generate(B, N, context, return_ppl=True)
        true_X = batch['X'].numpy()

        res_seqs, res_true_xs = [], []
        for i in range(len(seqs)):
            l, r = true_L[i].index(self.cdr_type), true_L[i].rindex(self.cdr_type)
            seq = seqs[i][:r - l + 1]
            res_seqs.append(seq)
            true_x = true_X[i][l:r+1]  # [N, 4, 3]
            res_true_xs.append(true_x)
        return ppls.detach().cpu().numpy().tolist(), res_seqs, res_true_xs, res_true_xs, True
