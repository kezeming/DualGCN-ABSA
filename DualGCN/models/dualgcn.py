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

    def forward(self, inputs):
        outputs1, outputs2, adj_ag, adj_dep = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2), dim=-1)
        logits = self.classifier(final_outputs)

        adj_ag_T = adj_ag.transpose(1, 2)
        identity = torch.eye(adj_ag.size(1)).cuda()
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        # A*A^T
        ortho = adj_ag @ adj_ag_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))  # 每个ortho正交矩阵的对角线元素置0
            ortho[i] += torch.eye(ortho[i].size(0)).cuda()  # 每个ortho正交矩阵的对角线元素加1

        # 根据loss类型设置正则化项？
        penal = None
        if self.opt.losstype == 'doubleloss':
            penal1 = (torch.norm(ortho - identity) / adj_ag.size(0)).cuda()
            penal2 = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).cuda()
            penal = self.opt.alpha * penal1 + self.opt.beta * penal2

        elif self.opt.losstype == 'orthogonalloss':
            penal = (torch.norm(ortho - identity) / adj_ag.size(0)).cuda()
            penal = self.opt.alpha * penal

        elif self.opt.losstype == 'differentiatedloss':
            penal = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).cuda()
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
            adj_dep = adj[:, :maxlen, :maxlen].float()
        else:
            def inputs_to_tree_reps(head, words, l):
                # Convert a sequence of head indexes into a tree object.
                trees = [head_to_tree(head[i], words[i], l[i]) for i in range(len(l))]
                # Convert a tree object to an (numpy) adjacency matrix.
                # directed参数：有向图 or 无向图
                # self_loop参数：是否自我可达，即i到i是否有边
                adj = [tree_to_adj(maxlen, tree, directed=self.opt.direct, self_loop=self.opt.loop).reshape(1, maxlen, maxlen) for tree in trees]
                adj = np.concatenate(adj, axis=0)
                adj = torch.from_numpy(adj)
                return adj.cuda()

            adj_dep = inputs_to_tree_reps(head.data, tok.data, l.data)

        h1, h2, adj_ag = self.gcn(adj_dep, inputs)
        # avg pooling asp feature
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num
        mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim)  # mask for h
        outputs1 = (h1 * mask).sum(dim=1) / asp_wn
        outputs2 = (h2 * mask).sum(dim=1) / asp_wn

        return outputs1, outputs2, adj_ag, adj_dep


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
        self.rnn_drop = nn.Dropout(opt.rnn_dropout)
        self.in_drop = nn.Dropout(opt.input_dropout)
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

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.opt.rnn_hidden, self.opt.rnn_layers, self.opt.bidirect)
        # pack_padded_sequence将填充过的数据进行压缩，避免填充的值对最终训练产生影响
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        tok, asp, pos, head, deprel, post, mask, l, _ = inputs  # unpack inputs
        src_mask = (tok != 0).unsqueeze(-2)
        maxlen = max(l.data)
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]

        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.opt.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        self.rnn.flatten_parameters()
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l, tok.size()[0]))

        denom_dep = adj.sum(2).unsqueeze(2) + 1
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        outputs_dep = None
        adj_ag = None

        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag /= self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        adj_ag = mask_ * adj_ag

        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_ag = gcn_inputs
        outputs_dep = gcn_inputs

        for l in range(self.layers):
            # ************SynGCN*************
            Ax_dep = adj.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            # ************SemGCN*************
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # * mutual Biaffine module
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)
            gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag

        return outputs_ag, outputs_dep, adj_ag


# 根据我们指定的[batch_size, hidden_dim, num_layers, bi-rnn], 生成一个双向rnn的初始h0和c0，并且把他们移到gpu中计算
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
            mask = mask.unsqueeze(1)  # 在第二维增加一个维度，比如[3, 2] => [3, 1, 2]

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
