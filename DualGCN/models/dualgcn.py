'''
Description: 
version: 
Author: chenhao
Date: 2021-06-09 14:17:37
'''
import copy
import logging
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree import head_to_tree, tree_to_adj


class DualGCNClassifier(nn.Module):
    # embedding就是一个查找表
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        in_dim = opt.hidden_dim
        self.opt = opt
        self.gcn_model = GCNAbsaModel(embedding_matrix=embedding_matrix, opt=opt)
        # 分类器
        self.classifier = nn.Linear(in_dim * 2, opt.polarities_dim)
        self.clr = nn.Linear(in_dim, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, outputs2, adj_sem, adj_syn = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2), dim=-1)  # [batch_size, 1, 2*mem_dim]
        logits = self.classifier(final_outputs)
        # outputs = torch.div(outputs1+outputs2, 2)
        # logits = self.clr(outputs1+outputs2)

        adj_sem_T = adj_sem.transpose(1, 2)
        identity = torch.eye(adj_sem.size(1)).cuda()
        # [batch_size, seq_len, seq_len]
        identity = identity.unsqueeze(0).expand(adj_sem.size(0), adj_sem.size(1), adj_sem.size(1))
        # A*A^T
        ortho = adj_sem @ adj_sem_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))  # 每个ortho正交矩阵的对角线元素置0
            ortho[i] += torch.eye(ortho[i].size(0)).cuda()  # 每个ortho正交矩阵的对角线元素置1

        # 根据loss类型设置正则化项
        # penal1 = R_O
        # penal2 = R_D
        penal = None
        if self.opt.losstype == 'doubleloss':
            penal1 = (torch.norm(ortho - identity) / adj_sem.size(0)).cuda()
            penal2 = (adj_sem.size(0) / torch.norm(adj_sem - adj_syn)).cuda()
            penal = self.opt.alpha * penal1 + self.opt.beta * penal2

        elif self.opt.losstype == 'orthogonalloss':
            penal = (torch.norm(ortho - identity) / adj_sem.size(0)).cuda()
            penal = self.opt.alpha * penal

        elif self.opt.losstype == 'differentiatedloss':
            penal = (adj_sem.size(0) / torch.norm(adj_sem - adj_syn)).cuda()
            penal = self.opt.beta * penal

        return logits, penal


class GCNAbsaModel(nn.Module):
    # embedding_matrix是我们初始时从glove模型中生成的emb查找表
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        self.embedding_matrix = embedding_matrix
        # 构建三个查找表emb、pos_emb、post_emb，其中emb训练过程中不更新，emb里存储的是glove模型中的embedding
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim,
                                     padding_idx=0) if opt.post_dim > 0 else None  # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)
        # gcn layer
        self.gcn = GCN(opt, embeddings, opt.hidden_dim, opt.num_layers)

    def forward(self, inputs):
        # token，aspect，pos，head，deprel，post，mask，length，adj
        tok, asp, pos, head, deprel, post, mask, l, adj = inputs  # unpack inputs
        maxlen = max(l.data)
        mask = mask[:, :maxlen]
        if self.opt.parseadj:
            adj_syn = adj[:, :maxlen, :maxlen].float()
        else:
            # 将输入序列中的head转换为tree，再转换为adj
            def inputs_to_tree_reps(head, words, l):
                # Convert a sequence of head indexes into a tree object.
                trees = [head_to_tree(head[i], words[i], l[i]) for i in range(len(l))]
                # Convert a tree object to an (numpy) adjacency matrix.
                # directed参数：有向图 or 无向图
                # self_loop参数：是否自我可达，即i到i是否有边
                adj = [tree_to_adj(maxlen, tree, directed=self.opt.direct, self_loop=self.opt.loop).reshape(1, maxlen, maxlen) for tree in trees]
                # [batch_size, maxlen, maxlen]
                adj = np.concatenate(adj, axis=0)
                adj = torch.from_numpy(adj)
                return adj.cuda()

            adj_syn = inputs_to_tree_reps(head.data, tok.data, l.data)

        h1, h2, adj_sem = self.gcn(adj_syn, inputs)
        # avg pooling asp feature
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num [batch_size, 1, 1]
        # [batch_size, maxlen, 1] => [batch_size, maxlen, hidden_dim]
        # 行重复hidden_dim次
        mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim)  # mask for h
        # [batch_size, seq_len, mem_dim] h1和mask同维
        outputs1 = (h1 * mask).sum(dim=1) / asp_wn  # ?
        outputs2 = (h2 * mask).sum(dim=1) / asp_wn

        # outputs1=outputs2=[batch_size, 1, mem_dim]
        # adj_sem=[batch_size, seq_len, seq_len]
        # adj_syn=[batch_size, maxlen, maxlen]
        return outputs1, outputs2, adj_sem, adj_syn


class GCN(nn.Module):
    # opt参数，embedding方式，GCN的隐藏层dim，层数
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim
        # 输入维度 = 单词embedding维度 + 位置维度 + 词性维度
        self.in_dim = opt.embed_dim + opt.post_dim + opt.pos_dim
        self.emb, self.pos_emb, self.post_emb = embeddings  # unpack

        # rnn layer
        input_size = self.in_dim
        # LSTM(input_size=360, hidden_size=50, nums_layer=1, input_format=[batch_size, seq_len, input_size(360)],
        # dropout=0.1, bidirectional=True)
        self.rnn = nn.LSTM(input_size, opt.rnn_hidden, opt.rnn_layers, batch_first=True,
                           dropout=opt.rnn_dropout, bidirectional=opt.bidirect)

        if opt.bidirect:
            self.in_dim = opt.rnn_hidden * 2  # 双向RNN将in_dim=改为二倍隐藏层维度
        else:
            self.in_dim = opt.rnn_hidden

        # 设置 drop out
        self.in_drop = nn.Dropout(opt.input_dropout)
        self.rnn_drop = nn.Dropout(opt.rnn_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()  # 存放多个子Module，子module自动注册到整个网络上，同时param也会添加到整个网络中
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        # attention 模块(只有一个注意力头)
        self.attention_heads = opt.attention_heads
        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim * 2)

        self.weight_list = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        # 双向交互模块，将SynGCN和SemGCN提取的特征进行交互
        # nn.Parameter将一个不可训练的tensor转换成可以训练的类型parameter
        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

        self.leakyrelu = nn.LeakyReLU(opt.gamma)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        # h0, c0=[total_layers, batch_size, hidden_dim]
        h0, c0 = rnn_zero_state(batch_size, self.opt.rnn_hidden, self.opt.rnn_layers, self.opt.bidirect)
        # pack_padded_sequence将填充过的数据进行压缩，避免填充的值对最终训练产生影响
        # seq_lens.cpu() 解决使用过高版本torch的报错
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens.cpu(), batch_first=True,
                                                       enforce_sorted=False)
        # rnn_outputs=[batch_size, seq_len, num_directions * hidden_size]
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        # *inputs中的所有数据的第二维都是 opt.max_len
        tok, asp, pos, head, deprel, post, mask, l, _ = inputs  # unpack inputs
        # [batch_size, max_len] => [batch_size, 1, max_len]
        src_mask = (tok != 0).unsqueeze(-2)
        maxlen = max(l.data)  # 此maxlen表示：在一个batch中，每条数据的实际token长度的最大值
        # *_like(W) 函数构建一个与W维度一致的矩阵，如zero_like、ones_like
        # [batch_size, max_len] => [batch_size, max_len, 1] => [batch_size, :maxlen, 1]
        # max_len=85代表sequence的最大长度, 而maxlen则是我们当前batch中实际token序列的最大长度
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]

        # embedding
        # [batch_size, max_len] => [batch_size, max_len, emb_dim]
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.opt.post_dim > 0:
            embs += [self.post_emb(post)]
        # embs = [batch_size, max_len, emb_dim] + [batch_size, max_len, pos_dim] + [batch_size, max_len, post_dim]
        # => [batch_size, max_len, emb_dim + pos_dim + post_dim]
        embs = torch.cat(embs, dim=2)
        # 设置embs的dropout
        embs = self.in_drop(embs)

        # rnn layer
        self.rnn.flatten_parameters()  # 重置参数数据指针，使用更快的代码路径
        # gcn_inputs=[batch_size, seq_len, num_directions * hidden_size]
        # 注意: 这里的 seq_len 其实就等于 maxlen
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l, tok.size()[0]))

        # adj已经在上层GCNAbsaModel中处理为长度为maxlen(有parseadj=True时)
        # adj=[batch_size, maxlen, maxlen] => [batch_size, maxlen] => [batch_size, maxlen, 1]
        # sum(2)是将adj的每一行进行相加：ai1+ai2+...+aiN = sum;
        denom_syn = adj.sum(2).unsqueeze(2) + 1
        # [batch_size, heads, seq_len, seq_len], attn_tensor是注意力权重
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        # attn = [batch_size, heads, seq_len, seq_len]
        # =>[heads, batch_size, seq_len, seq_len]
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        outputs_syn = None
        adj_sem = None

        # * Average Multi-head Attention matrices
        # * 将多头注意力矩阵相加，然后做平均
        for i in range(self.attention_heads):
            if adj_sem is None:
                adj_sem = attn_adj_list[i]
            else:
                adj_sem += attn_adj_list[i]  # 矩阵对应位置值直接相加
        # adj_sem=[batch_size, seq_len, seq_len]
        adj_sem = adj_sem / self.attention_heads  # bug fix！

        for j in range(adj_sem.size(0)):
            adj_sem[j] -= torch.diag(torch.diag(adj_sem[j]))  # 对角线上的值置为0
            adj_sem[j] += torch.eye(adj_sem[j].size(0)).cuda()  # 将对角线上值置为1
        # [batch_size, maxlen, 1] * [batch_size, seq_len, seq_len]  seq_len 其实就等于 maxlen
        # => [batch_size, seq_len, seq_len]
        adj_sem = mask_ * adj_sem

        # [batch_size, seq_len, seq_len] => [batch_size, seq_len] => [batch_size, seq_len, 1]
        denom_sem = adj_sem.sum(2).unsqueeze(2) + 1
        outputs_sem = gcn_inputs
        outputs_syn = gcn_inputs

        for layer in range(self.layers):
            # ************SynGCN*************
            #             基于语法 --dep
            # adj=[batch_size, maxlen, maxlen]
            # outputs_syn=[batch_size, seq_len, num_directions * hidden_size]
            Ax_syn = adj.bmm(outputs_syn)  # [batch_size, seq_len, num_directions*hidden_size]
            AxW_syn = self.W[layer](Ax_syn)  # [batch_size, seq_len, mem_dim]
            AxW_syn = AxW_syn / denom_syn
            # H_syn = F.relu(AxW_syn)
            H_syn = self.leakyrelu(AxW_syn)

            # ************SemGCN*************
            #             基于语义 --ag
            # adj_sem=[batch_size, seq_len, seq_len]
            # outputs_sem=[batch_size, seq_len, num_directions * hidden_size]
            Ax_sem = adj_sem.bmm(outputs_sem)
            AxW_sem = self.weight_list[layer](Ax_sem)
            AxW_sem = AxW_sem / denom_sem
            # H_sem = F.relu(AxW_sem)
            H_sem = self.leakyrelu(AxW_sem)

            # * mutual Biaffine module
            # [batch_size, seq_len, seq_len]
            A1 = F.softmax(torch.bmm(torch.matmul(H_syn, self.affine1), torch.transpose(H_sem, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(H_sem, self.affine2), torch.transpose(H_syn, 1, 2)), dim=-1)
            # H_syn_prime=H_syn' [batch_size, seq_len, mem_dim]
            # H_sem_prime=H_sem' [batch_size, seq_len, mem_dim]
            H_syn_prime, H_sem_prime = torch.bmm(A1, H_sem), torch.bmm(A2, H_syn)
            outputs_syn = self.gcn_drop(H_syn_prime) if layer < self.layers - 1 else H_syn_prime
            outputs_sem = self.gcn_drop(H_sem_prime) if layer < self.layers - 1 else H_sem_prime

        return outputs_sem, outputs_syn, adj_sem


# 根据我们指定的[batch_size, hidden_dim, num_layers, bi-rnn]
# 根据rnn的层数以及是否双向生成rnn的初始的h0和c0，并且把他们移到gpu中计算
def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    # Variable 本质上就一个tensor，只不过在其基础上封装了一些其他的属性
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    # 这里将初始隐藏态h0和初始元胞态c0移到gpu中计算
    return h0.cuda(), c0.cuda()


# 根据query，key计算注意力权重
def attention(query, key, mask=None, dropout=None):
    # query = [batch_size, heads, seq_len, d_k], d_k = d_model // h
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # scores = [batch_size, heads, seq_len, seq_len]
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        # 这里会打破之前算的softmax概率，没有影响吗？
        p_attn = dropout(p_attn)

    # 返回的是注意力权重p_attn = [batch_size, heads, seq_len, seq_len]
    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    # heads, mem_dim*2 (隐藏层维度*2), dropout
    def __init__(self, head, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadAttention, self).__init__()
        assert d_model % head == 0, "d_model % h != 0 ERROR"
        self.d_k = d_model // head
        self.head = head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 注意这里就声明了两个的全连接层
        self.dropout = nn.Dropout(p=dropout)

    # 注意这里传进来的query和key都是一样的，都等于gcn_inputs
    def forward(self, query, key, mask=None):
        # 注意这里的mask，他是从Instructor->DualGCNClassifier->GCNAbsaModel->GCN->MultiHeadAttention
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len] => [batch_size, 1, 1, max_len]

        batch_size = query.size(0)  # query[batch_size, seq_len, d_model]

        # 维度变化情况
        # 原始query、key = [batch_size, seq_len, d_model]，经过linear[全连接]线性层变换后维度不变
        # 等价于query=linear(query), key=linear(key), value=linear(value)
        # 经过view和transpose变换后从 [batch_size, seq_len, d_model] => [batch_size, heads, seq_len, d_k]
        query, key = [l(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]  # 因为只声明了两个linear，所以这里只传query和key

        # attn = [batch_size, heads, seq_len, seq_len]
        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn
