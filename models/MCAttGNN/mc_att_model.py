#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch import masked_fill, nn
import torch.nn.functional as F

from torch_scatter import scatter_sum

from data import VOCAB

from .mc_egnn import MCAttEGNN, MCEGNN, PureMCAtt


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


class ProteinFeature(nn.Module):

    def __init__(self):
        super().__init__()
        # global nodes and mask nodes
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)

        # segment ids
        self.ag_seg_id, self.hc_seg_id, self.lc_seg_id = 1, 2, 3

    def _is_global(self, S):
        return sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)  # [N]

    def _construct_segment_ids(self, S):
        # construct segment ids. 1/2/3 for antigen/heavy chain/light chain
        glbl_node_mask = self._is_global(S)
        glbl_nodes = S[glbl_node_mask]
        boa_mask, boh_mask, bol_mask = (glbl_nodes == self.boa_idx), (glbl_nodes == self.boh_idx), (glbl_nodes == self.bol_idx)
        glbl_nodes[boa_mask], glbl_nodes[boh_mask], glbl_nodes[bol_mask] = self.ag_seg_id, self.hc_seg_id, self.lc_seg_id
        segment_ids = torch.zeros_like(S)
        segment_ids[glbl_node_mask] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        segment_ids = torch.cumsum(segment_ids, dim=0)
        return segment_ids

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id, segment_ids=None):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # prepare inputs
        if segment_ids is None:
            segment_ids = self._construct_segment_ids(S)

        ctx_edges, inter_edges = [], []

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        
        # not global edges
        is_global = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx) # [N]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))

        # all possible ctx edges: same seg, not global
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        ctx_edges = _radial_edges(X, torch.stack([ctx_all_row, ctx_all_col]).T, cutoff=8.0)

        # all possible inter edges: not same seg, not global
        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        inter_edges = _radial_edges(X, torch.stack([inter_all_row, inter_all_col]).T, cutoff=12.0)

        # edges between global and normal nodes
        select_edges = torch.logical_and(row_seg == col_seg, torch.logical_not(not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = torch.logical_and(row_global, col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]

        # add additional edge to neighbors in 1D sequence (except epitope)
        select_edges = sequential_and(
            torch.logical_or((row - col) == 1, (row - col) == -1),  # adjacent in the graph
            not_global_edges,  # not global edges (also ensure the edges are in the same segment)
            row_seg != self.ag_seg_id  # not epitope
        )
        seq_adj = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]

        # finally construct context edges
        space_edge_num = ctx_edges.shape[1] + global_normal.shape[1] + global_global.shape[1]
        ctx_edges = torch.cat([ctx_edges, global_normal, global_global, seq_adj], dim=1)  # [2, E]
        ctx_edge_feats = torch.cat(
            [torch.zeros(space_edge_num, dtype=torch.float, device=X.device), 
             torch.ones(seq_adj.shape[1], dtype=torch.float, device=X.device)], dim=0).unsqueeze(-1)

        return ctx_edges, inter_edges, ctx_edge_feats

    def forward(self, X, S, offsets):
        batch_id = torch.zeros_like(S)
        batch_id[offsets[1:-1]] = 1
        batch_id.cumsum_(dim=0)

        return self.construct_edges(X, S, batch_id)


def _radial_edges(X, src_dst, cutoff):
    dist = X[:, 1][src_dst]  # [Ef, 2, 3], CA position
    dist = torch.norm(dist[:, 0] - dist[:, 1], dim=-1) # [Ef]
    src_dst = src_dst[dist <= cutoff]
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]
    return src_dst


class MCAttModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, n_edge_feats=0,
                 n_layers=3, cdr_type='3', alpha=0.1, dropout=0.1, dense=False):
        super().__init__()
        self.num_aa_type = len(VOCAB)
        self.cdr_type = cdr_type
        self.mask_token_id = VOCAB.get_unk_idx()
        self.alpha = alpha

        self.aa_embedding = nn.Embedding(self.num_aa_type, embed_size)
        self.gnn = MCAttEGNN(embed_size, hidden_size, self.num_aa_type,
                             n_channel, n_edge_feats, n_layers=n_layers,
                             residual=True, dropout=dropout, dense=dense)
        
        # self.seq_loss = nn.CrossEntropyLoss(reduction='none')
        # self.coord_loss = nn.MSELoss(reduction='sum')
        # it is the same as HuberLoss with delta=1.0
        # self.coord_loss = nn.SmoothL1Loss(reduction='sum')
        
        # edges
        self.protein_feature = ProteinFeature()

    def seq_loss(self, _input, target):
        return F.cross_entropy(_input, target, reduction='none')

    def coord_loss(self, _input, target):
        return F.smooth_l1_loss(_input, target, reduction='sum')

    def init_mask(self, X, S, cdr_range):
        '''
        set coordinates of masks following a unified distribution
        between the two ends
        '''
        X, S, cmask = X.clone(), S.clone(), torch.zeros_like(X, device=X.device)
        n_channel, n_dim = X.shape[1:]
        for start, end in cdr_range:
            S[start:end + 1] = self.mask_token_id
            l_coord, r_coord = X[start - 1], X[end + 1]  # [n_channel, 3]
            n_span = end - start + 2
            coord_offsets = (r_coord - l_coord).unsqueeze(0).expand(n_span - 1, n_channel, n_dim)  # [n_mask, n_channel, 3]
            coord_offsets = torch.cumsum(coord_offsets, dim=0)
            mask_coords = l_coord + coord_offsets / n_span
            X[start:end + 1] = mask_coords
            cmask[start:end + 1, ...] = 1
        return X, S, cmask
    

    def forward(self, X, S, L, offsets):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param L: list of cdr types
        :param offsets: [batch_size + 1]
        '''
        # prepare inputs
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)
        max_step = max([r - l + 1 for l, r in cdr_range])
        pred_positions, aa_cnt = [], 0
        for t in range(max_step):
            position = cdr_range[:, 0] + t
            position = position.masked_select(position <= cdr_range[:, 1])
            pred_positions.append(position)
            aa_cnt += len(position)

        # save ground truth
        true_X, true_S = X.clone(), S.clone()

        # init mask
        X, S, cmask = self.init_mask(X, S, cdr_range)

        snll, closs = 0, 0
        for t in range(max_step):
            with torch.no_grad():
                ctx_edges, inter_edges, ctx_edge_feats = self.protein_feature(X, S, offsets)

            pred_position = pred_positions[t]

            H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
            H, Z = self.gnn(H_0, X, ctx_edges, inter_edges, ctx_edge_feats)
            
            # loss
            logits = H[pred_position]
            gt_s, gt_x = true_S[pred_position], true_X[pred_position]
            snll = snll + torch.sum(self.seq_loss(logits, gt_s)) / aa_cnt
            # closs = closs + self.coord_loss(Z[pred_position], gt_x) / aa_cnt
            cur_cmask = torch.zeros_like(cmask, device=cmask.device, dtype=torch.bool)
            cur_cmask[pred_position] = 1
            closs = closs + self.coord_loss(Z.masked_select(cur_cmask), true_X.masked_select(cur_cmask)) / aa_cnt
            
            # update X, S
            S = S.clone()
            S[pred_position] = true_S[pred_position]
            # pure autoregressive
            # cmask = cmask.clone()
            # cmask[pred_position] = 0  # use ground truth
            X = (1 - cmask) * true_X + cmask * Z

        loss = snll + self.alpha * closs
        return loss, snll, closs


    def generate(self, X, S, L, offsets, greedy=True):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param L: list of cdr types
        :param offsets: [batch_size + 1]
        '''
        # prepare inputs
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)
        max_step = max([r - l + 1 for l, r in cdr_range])
        pred_positions, aa_cnt = [], 0
        for t in range(max_step):
            position = cdr_range[:, 0] + t
            position = position.masked_select(position <= cdr_range[:, 1])
            pred_positions.append(position)
            aa_cnt += len(position)

        # save ground truth
        true_X, true_S = X.clone(), S.clone()

        # init mask
        X, S, cmask = self.init_mask(X, S, cdr_range)

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)

        snll_all_step = []
        for t in range(max_step):
            with torch.no_grad():
                ctx_edges, inter_edges, ctx_edge_feats = self.protein_feature(X, S, offsets)

            pred_position = pred_positions[t]

            H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
            H, Z = self.gnn(H_0, X, ctx_edges, inter_edges, ctx_edge_feats)
            
            # sequence
            smask = special_mask.repeat(len(pred_position), 1).bool()  # [n, vocab_size]
            logits = H[pred_position]
            logits = logits.masked_fill(smask, float('-inf'))  # mask special tokens
            if greedy:
                S[pred_position] = torch.argmax(logits, dim=-1)  # [n]
            else:
                prob = F.softmax(logits, dim=-1)  # [n, vocab_size]
                S[pred_position] = torch.multinomial(prob, num_samples=1).squeeze()

            # nll loss
            snll_all_step.append(self.seq_loss(logits, S[pred_position]))
            
            # update X
            # cmask = cmask.clone()
            # cmask[pred_position] = 0  # fix already predicted X
            X = (1 - cmask) * X + cmask * Z

        return snll_all_step, S, X, true_X, cdr_range

    # unified api: return ppl (list), seq (list), x (numpy), true_x (numpy) and whether aligned
    def infer(self, batch, device):
        X, S, L, offsets = batch['X'].to(device), batch['S'].to(device), batch['L'], batch['offsets'].to(device)
        snll_all_step, pred_S, pred_X, true_X, cdr_range = self.generate(
            X, S, L, offsets
        )
        pred_S, cdr_range = pred_S.tolist(), cdr_range.tolist()
        pred_X, true_X = pred_X.cpu().numpy(), true_X.cpu().numpy()
        # seqs, x, true_x
        seq, x, true_x = [], [], []
        for start, end in cdr_range:
            end = end + 1
            seq.append(''.join([VOCAB.idx_to_symbol(pred_S[i]) for i in range(start, end)]))
            x.append(pred_X[start:end])
            true_x.append(true_X[start:end])
        # ppl
        ppl = [0 for _ in range(len(cdr_range))]
        lens = [0 for _ in ppl]
        for t, snlls in enumerate(snll_all_step):
            i, j = 0, 0
            if len(snlls.shape) == 0:
                snlls = snlls.reshape(1)
            for start, end in cdr_range:
                if start + t <= end:
                    ppl[i] += snlls[j]
                    lens[i] += 1
                    j += 1
                i += 1
        ppl = [p / n for p, n in zip(ppl, lens)]
        ppl = torch.exp(torch.tensor(ppl, device=device)).tolist()
        return ppl, seq, x, true_x, True


class EfficientMCAttModel(MCAttModel):
    def __init__(self, embed_size, hidden_size, n_channel, n_edge_feats=0, n_layers=3, cdr_type='3', alpha=0.1, dropout=0.1, n_iter=5):
        super().__init__(embed_size, hidden_size, n_channel, n_edge_feats, n_layers, cdr_type, alpha, dropout, dense=False)
        self.n_iter = n_iter

    def forward(self, X, S, L, offsets):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param L: list of cdr types
        :param offsets: [batch_size + 1]
        '''
        # prepare inputs
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        # save ground truth
        true_X, true_S = X.clone(), S.clone()

        # init mask
        X, S, cmask = self.init_mask(X, S, cdr_range)  # [n_all_node, n_channel, 3]
        mask = cmask[:, 0, 0].bool()  # [n_all_node]
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()
        H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
        aa_embeddings = self.aa_embedding(torch.arange(self.num_aa_type, device=H_0.device))  # [vocab_size, embed_size]

        snll, closs = 0, 0

        for r in range(self.n_iter):
            with torch.no_grad():
                ctx_edges, inter_edges, ctx_edge_feats = self.protein_feature(X, S, offsets)
            H, Z = self.gnn(H_0, X, ctx_edges, inter_edges, ctx_edge_feats)

            # refine
            X = X.clone()
            X[mask] = Z[mask]
            H_0 = H_0.clone()
            logits = H[mask]
            seq_prob = torch.softmax(logits.masked_fill(smask, float('-inf')), dim=-1)  # [aa_cnt, vocab_size]
            H_0[mask] = seq_prob.mm(aa_embeddings)  # smooth embedding
            
            r_snll = torch.sum(self.seq_loss(logits, true_S[mask])) / aa_cnt
            snll += r_snll / self.n_iter
        
        closs = self.coord_loss(Z[mask], true_X[mask]) / aa_cnt

        loss = snll + self.alpha * closs
        return loss, r_snll, closs  # only return the last snll

    def generate(self, X, S, L, offsets, greedy=True):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param L: list of cdr types
        :param offsets: [batch_size + 1]
        '''
        # prepare inputs
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        # save ground truth
        true_X, true_S = X.clone(), S.clone()

        # init mask
        X, S, cmask = self.init_mask(X, S, cdr_range)
        mask = cmask[:, 0, 0].bool()  # [n_all_node]
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()
        H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
        aa_embeddings = self.aa_embedding(torch.arange(self.num_aa_type, device=H_0.device))  # [vocab_size, embed_size]
        
        for r in range(self.n_iter):
            with torch.no_grad():
                ctx_edges, inter_edges, ctx_edge_feats = self.protein_feature(X, S, offsets)
            H, Z = self.gnn(H_0, X, ctx_edges, inter_edges, ctx_edge_feats)

            # refine
            X = X.clone()
            X[mask] = Z[mask]
            H_0 = H_0.clone()
            seq_prob = torch.softmax(H[mask].masked_fill(smask, float('-inf')), dim=-1)  # [aa_cnt, vocab_size]
            H_0[mask] = seq_prob.mm(aa_embeddings)  # smooth embedding

        # H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
        # H, Z = self.gnn(H_0, X, ctx_edges, inter_edges, ctx_edge_feats)

        logits = H[mask]  # [aa_cnt, vocab_size]

        logits = logits.masked_fill(smask, float('-inf'))  # mask special tokens

        if greedy:
            S[mask] = torch.argmax(logits, dim=-1)  # [n]
        else:
            prob = F.softmax(logits, dim=-1)
            S[mask] = torch.multinomial(prob, num_samples=1).squeeze()
        snll_all = self.seq_loss(logits, S[mask])

        return snll_all, S, X, true_X, cdr_range

    def infer(self, batch, device, greedy=True):
        X, S, L, offsets = batch['X'].to(device), batch['S'].to(device), batch['L'], batch['offsets'].to(device)
        snll_all, pred_S, pred_X, true_X, cdr_range = self.generate(
            X, S, L, offsets, greedy=greedy
        )
        pred_S, cdr_range = pred_S.tolist(), cdr_range.tolist()
        pred_X, true_X = pred_X.cpu().numpy(), true_X.cpu().numpy()
        # seqs, x, true_x
        seq, x, true_x = [], [], []
        for start, end in cdr_range:
            end = end + 1
            seq.append(''.join([VOCAB.idx_to_symbol(pred_S[i]) for i in range(start, end)]))
            x.append(pred_X[start:end])
            true_x.append(true_X[start:end])
        # ppl
        ppl = [0 for _ in range(len(cdr_range))]
        lens = [0 for _ in ppl]
        offset = 0
        for i, (start, end) in enumerate(cdr_range):
            length = end - start + 1
            for t in range(length):
                ppl[i] += snll_all[t + offset]
            offset += length
            lens[i] = length
        ppl = [p / n for p, n in zip(ppl, lens)]
        ppl = torch.exp(torch.tensor(ppl, device=device)).tolist()
        return ppl, seq, x, true_x, True

    def generate_analyze(self, X, S, L, offsets, greedy=True):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param L: list of cdr types
        :param offsets: [batch_size + 1]
        '''
        # prepare inputs
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        # save ground truth
        true_X, true_S = X.clone(), S.clone()

        # init mask
        X, S, cmask = self.init_mask(X, S, cdr_range)
        mask = cmask[:, 0, 0].bool()  # [n_all_node]
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()
        H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
        aa_embeddings = self.aa_embedding(torch.arange(self.num_aa_type, device=H_0.device))  # [vocab_size, embed_size]
        
        round_probs, round_Xs = [], []

        for r in range(self.n_iter):
            with torch.no_grad():
                ctx_edges, inter_edges, ctx_edge_feats = self.protein_feature(X, S, offsets)
            H, Z, atts = self.gnn(H_0, X, ctx_edges, inter_edges, ctx_edge_feats, return_attention=True)

            # refine
            X = X.clone()
            X[mask] = Z[mask]
            H_0 = H_0.clone()
            masked_logits = H[mask].masked_fill(smask, float('-inf'))
            seq_prob = torch.softmax(masked_logits, dim=-1)  # [aa_cnt, vocab_size]
            H_0[mask] = seq_prob.mm(aa_embeddings)  # smooth embedding

            round_probs.append(masked_logits)
            round_Xs.append(X)

        # H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
        # H, Z = self.gnn(H_0, X, ctx_edges, inter_edges, ctx_edge_feats)

        logits = H[mask]  # [aa_cnt, vocab_size]

        logits = logits.masked_fill(smask, float('-inf'))  # mask special tokens

        if greedy:
            S[mask] = torch.argmax(logits, dim=-1)  # [n]
        else:
            prob = F.softmax(logits, dim=-1)
            S[mask] = torch.multinomial(prob, num_samples=1).squeeze()  # [n]
        snll_all = self.seq_loss(logits, S[mask])

        return {
            'r_probs': round_probs,
            'r_Xs': round_Xs,
            'cdr_range': cdr_range,
            'final_atts': atts
        }


class EfficientPureMCAttModel(EfficientMCAttModel):
    def __init__(self, embed_size, hidden_size, n_channel, n_edge_feats=0, n_layers=3, cdr_type='3', alpha=0.1, dropout=0.1):
        super().__init__(embed_size, hidden_size, n_channel, n_edge_feats, n_layers, cdr_type, alpha, dropout)
        self.gnn = PureMCAtt(embed_size, hidden_size, self.num_aa_type,
                             n_channel, n_edge_feats, n_layers=n_layers,
                             residual=True, dropout=dropout)

# these are for ablation study. (without interface attention layer)
class MCEGNNModel(MCAttModel):
    def __init__(self, embed_size, hidden_size, n_channel, n_edge_feats=0, n_layers=3, cdr_type='3', alpha=0.1, dropout=0.1):
        super().__init__(embed_size, hidden_size, n_channel, n_edge_feats, n_layers, cdr_type, alpha, dropout)
        self.gnn = MCEGNN(embed_size, hidden_size, self.num_aa_type,
                          n_channel, n_edge_feats, n_layers=n_layers,
                          residual=True, dropout=dropout)

class EfficientMCEGNNModel(EfficientMCAttModel):
    def __init__(self, embed_size, hidden_size, n_channel, n_edge_feats=0, n_layers=3, cdr_type='3', alpha=0.1, dropout=0.1, n_iter=5):
        super().__init__(embed_size, hidden_size, n_channel, n_edge_feats, n_layers, cdr_type, alpha, dropout, n_iter)
        self.gnn = MCEGNN(embed_size, hidden_size, self.num_aa_type,
                          n_channel, n_edge_feats, n_layers=n_layers,
                          residual=True, dropout=dropout)