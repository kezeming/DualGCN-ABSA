'''
Description: 
version: 
Author: chenhao
Date: 2021-06-09 14:17:37
'''
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class DualGCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim*2, opt.polarities_dim)
        in_dim = opt.hidden_dim

        # 线性变换
        self.linear_transfor = nn.ModuleList()
        for _ in range(2):
            self.linear_transfor.append(nn.Linear(in_dim // 2, in_dim // 2))

        # 分类器
        self.classifier = nn.Linear(in_dim, opt.polarities_dim)

        # Pyramid Layer
        self.input_dim = in_dim
        self.pyramid_layer = nn.ModuleList()
        total_dim = 0
        for _ in range(opt.pyramid):
            total_dim += self.input_dim
            self.pyramid_layer.append(nn.Linear(self.input_dim * 2, self.input_dim))
            self.input_dim = self.input_dim // 2

        self.pyramid_alaph = nn.Linear(total_dim, total_dim)
        self.W_r = nn.Linear(total_dim, in_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        outputs1, outputs2, adj_sem, adj_syn, pooled_output = self.gcn_model(inputs)

        # 线性变换
        outputs1 = self.opt.alpha * self.linear_transfor[0](outputs1)
        outputs2 = self.opt.beta * self.linear_transfor[1](outputs2)

        final_outputs = torch.cat((outputs1, outputs2), dim=-1)  # [batch_size, 1, 2*mem_dim]

        # Pyramid Layer Output
        all_outputs = None
        current_output = final_outputs
        for layer in range(self.opt.pyramid):
            next_output = self.pyramid_layer[layer](current_output)
            if all_outputs is None:
                all_outputs = next_output
            else:
                all_outputs = torch.cat((all_outputs, next_output), dim=-1)
            current_output = next_output
        fin_outputs = self.tanh(self.W_r(self.pyramid_alaph(all_outputs)))
        logits = self.classifier(fin_outputs)

        penal = (adj_sem.size(0) / torch.norm(adj_sem - adj_syn)).cuda()
        
        return logits, penal


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
        h1, h2, adj_ag, pooled_output = self.gcn(adj_dep, inputs)
        
        # avg pooling asp feature
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2) 
        outputs1 = (h1*aspect_mask).sum(dim=1) / asp_wn
        outputs2 = (h2*aspect_mask).sum(dim=1) / asp_wn
        return outputs1, outputs2, adj_ag, adj_dep, pooled_output


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.leakyrelu = nn.LeakyReLU(opt.gamma)

    def forward(self, adj, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
        src_mask = src_mask.unsqueeze(-2)
        
        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        denom_syn = adj.sum(2).unsqueeze(2) + 1
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        multi_head_list = []
        outputs_syn = None
        adj_sem = None
        
        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_sem is None:
                adj_sem = attn_adj_list[i]
            else:
                adj_sem += attn_adj_list[i]
        adj_sem = adj_sem / self.attention_heads  # bug fix！

        for j in range(adj_sem.size(0)):
            adj_sem[j] -= torch.diag(torch.diag(adj_sem[j]))
            adj_sem[j] += torch.eye(adj_sem[j].size(0)).cuda()
        adj_sem = src_mask.transpose(1, 2) * adj_sem

        denom_sem = adj_sem.sum(2).unsqueeze(2) + 1
        H_syn = None
        H_sem = None
        outputs_sem = gcn_inputs
        outputs_syn = gcn_inputs

        for l in range(self.layers):
            # ************SynGCN*************
            Ax_syn = adj.bmm(outputs_syn)
            AxW_syn = self.W[l](Ax_syn)
            AxW_syn = AxW_syn / denom_syn
            # gAxW_dep = F.relu(AxW_dep)
            H_syn = self.leakyrelu(AxW_syn)

            # ************SemGCN*************
            Ax_sem = adj_sem.bmm(outputs_sem)
            AxW_sem = self.weight_list[l](Ax_sem)
            AxW_sem = AxW_sem / denom_sem
            # gAxW_ag = F.relu(AxW_ag)
            H_sem = self.leakyrelu(AxW_sem)

            outputs_syn = self.gcn_drop(H_syn) if l < self.layers - 1 else H_syn
            outputs_sem = self.gcn_drop(H_sem) if l < self.layers - 1 else H_sem

        return H_syn, H_sem, adj_sem, pooled_output


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn