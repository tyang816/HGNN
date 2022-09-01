import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GatedGraphConv, GATConv, global_max_pool
from torch_geometric.utils import add_self_loops, softmax
from src.Utils import sequence_mask, tuple_map
import math
from collections import namedtuple


class GlobalAttention(nn.Module):
    ''' Multiplicative and additive global attention.
        shape of query: [batch_size, units]
        shape of key: [batch_size, max_steps, key_dim]
        shape of context: [batch_size, units]
        shape of alignments: [batch_size, max_steps]
        style should be either "add" or "mul"'''

    def __init__(self, units, key_dim=None, style='mul', scale=True):
        super(GlobalAttention, self).__init__()
        self.style = style
        self.scale = scale
        key_dim = key_dim or units

        self.Wk = nn.Linear(key_dim, units, bias=False)
        if self.style == 'mul':
            if self.scale:
                self.v = nn.Parameter(torch.tensor(1.))
        elif self.style == 'add':
            self.Wq = nn.Linear(units, units)
            self.v = nn.Parameter(torch.ones(units))
        else:
            raise ValueError(str(style) + ' is not an appropriate attention style.')

    def score(self, query, key):
        query = query.unsqueeze(1)  # batch_size * 1 * units
        key = self.Wk(key)

        if self.style == 'mul':
            output = torch.bmm(query, key.transpose(1, 2))
            output = output.squeeze(1)
            if self.scale:
                output = self.v * output
        else:
            output = torch.sum(self.v * torch.tanh(self.Wq(query) + key), 2)
        return output

    def forward(self, query, key, memory_lengths=None, custom_mask=None):
        score = self.score(query, key)  # batch_size * max_steps
        if memory_lengths is not None:
            score_mask = sequence_mask(memory_lengths, key.shape[1])
            score.masked_fill_(~score_mask, float('-inf'))
        elif custom_mask is not None:
            score.masked_fill_(~custom_mask, float('-inf'))

        alignments = F.softmax(score, 1)
        context = torch.bmm(alignments.unsqueeze(1), key)  # batch_size * 1 * units
        context = context.squeeze(1)
        return context, alignments


class StaticMsgPass(MessagePassing):
    def __init__(self):
        super(StaticMsgPass, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        # x: [N, in_dim]
        # edge_index: [2, E]
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        out = self.propagate(edge_index=edge_index, x=x)
        return out


class DynamicMsgPass(MessagePassing):
    def __init__(self, hidden_dim):
        super(DynamicMsgPass, self).__init__(aggr='add')
        self.Relu = nn.ReLU()
        self.WQ = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WK = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WR = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WV = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WF = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, edge_index, edge_attr, ptr):
        node_num = x.shape[0]
        edge_num = edge_index.shape[1]
        # x: [node_num, hidden_dim], edge_index: [2, edge_num], edge_attr: [edge_num, edge_dim]
        q = self.Relu(self.WQ(x))
        # k: [node_num, node_num, hidden_dim]
        k = self.Relu(self.WK(x)).repeat(node_num, 1, 1)
        e = self.Relu(self.WR(edge_attr))
        row, col = edge_index
        for src, trg, edge in zip(row, col, e):
            k[int(src), int(trg), :] = k[int(src), int(trg), :] + edge
        A = torch.zeros((node_num, node_num))
        idx = 0
        start = int(ptr[idx])
        end = int(ptr[idx + 1])
        d = q.shape[1]
        for node_id in range(node_num):
            # [1, hidden_dim] x [hidden_dim, node_num]
            A[node_id, start: end] = (q[node_id].unsqueeze(0) @ k[node_id].transpose(0, 1)).squeeze(0)[
                                     start: end] / math.sqrt(d)
            A[node_id, start: end] = F.softmax(A[node_id, start: end], dim=-1)
            if node_id > end:
                idx = idx + 1
                start = int(ptr[idx])
                end = int(ptr[idx + 1])
        A_dyn = torch.zeros((edge_num, 1))
        for edge_id in range(len(row)):
            A_dyn[edge_id] = A[int(row[edge_id]), int(col[edge_id])]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=A_dyn)
        return out

    def message(self, x_j, edge_attr, norm):
        # h: [edge_num, hidden_dim]
        h = self.WV(x_j)
        e = self.WF(edge_attr)
        return norm * (h + e)


class HybridMsgPass(nn.Module):
    def __init__(self, hidden_dim):
        super(HybridMsgPass, self).__init__()
        self.GRU = nn.GRUCell(hidden_dim, hidden_dim)
        self.Sigmoid = nn.Sigmoid()
        self.W_z = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, f, h_v, h_v_):
        # h_v/h_v_: [node_num, hidden_dim]
        # f: [node_num, hidden_dim]
        z_ = torch.cat((h_v, h_v_, h_v - h_v_, h_v * h_v_), dim=-1)
        # z: [node_num, hidden_dim]
        z = self.Sigmoid(self.W_z(z_))
        # fuse: [node_num, hidden_dim]
        fuse = z * h_v + (1 - z) * h_v_
        node_num = h_v.shape[0]
        f_k = torch.zeros_like(h_v)
        for idx in range(node_num):
            f_k[idx] = self.GRU(f[idx].unsqueeze(0), fuse[idx].unsqueeze(0))
        return f_k


class Encoder(nn.Module):
    def __init__(self, code_embed_dim, com_embed_dim, edge_embed_dim, hidden_dim, hops,
                 code_field=None, com_field=None):
        super(Encoder, self).__init__()
        assert code_field is not None and com_field is not None
        code_vocab_size = len(code_field.vocab)
        com_vocab_size = len(com_field.vocab)
        pad_id = code_field.vocab.stoi['<pad>']
        self.code_field = code_field
        self.com_field = com_field
        self.hops = hops
        self.hidden_dim = hidden_dim
        self.CodeEmbedding = nn.Embedding(code_vocab_size, code_embed_dim, padding_idx=pad_id)
        self.BiLSTM = nn.LSTM(hidden_dim, 128, bidirectional=True)
        self.ComEembedding = nn.Embedding(com_vocab_size, com_embed_dim)
        self.EdgeEmbedding = nn.Embedding(2, edge_embed_dim)
        self.W_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.W_z = nn.Linear(4 * hidden_dim, hidden_dim)
        self.Relu = nn.ReLU()

    def forward(self, src, rev, sim_score):
        # src_node: [N, node_dim], src_edge: [2, edge_num], src_attr: [edge_num], y: [com_len]
        src_node, src_edge, src_attr, src_com = src.x, src.edge_index, src.edge_attr, src.y
        rev_node, rev_edge, rev_attr, rev_com = rev.x, rev.edge_index, rev.edge_attr, rev.y
        batch = src.batch
        ptr = src.ptr
        batch_size = len(ptr) - 1
        node_num = src_node.shape[0]
        # print(ptr, src_node.shape)
        # Graph Initialization
        # initialization representation
        H_c = self.CodeEmbedding(src_node).squeeze(1)
        H_c_ = self.CodeEmbedding(rev_node).squeeze(1)
        E_c = self.EdgeEmbedding(src_attr)

        # Retrieval-Based Augmentation
        # Retrieved Code-based Augmentation
        # A_aug: [N, N]
        A_aug = torch.exp(self.Relu(self.W_C(H_c)) @ self.Relu(self.W_Q(H_c_)).transpose(0, 1))
        H1_c = torch.zeros((node_num, self.hidden_dim))
        H_aug = A_aug @ H_c_
        for graph_id in range(batch_size):
            start = ptr[graph_id]
            end = ptr[graph_id + 1]
            H1_c[start: end] = sim_score[graph_id] * H_aug[start: end]
        # comp: [N, hidden_dim]
        comp = H_c + H1_c
        # Retrieved Summary-based Augmentation
        # rev_com: [batch_size, com_len]
        rev_com = torch.reshape(rev_com, (batch_size, -1))
        com_embed = self.ComEembedding(rev_com)
        out, _ = self.BiLSTM(com_embed)
        hn = out[:,-1,:]
        h = torch.tensor(sim_score).view(-1, 1, 1) * out

        # Hybrid GNN
        # Static Message Passing
        smp = StaticMsgPass()
        # h_v: [N, hidden_dim]
        h_v = comp
        for _ in range(self.hops):
            h_v = smp(h_v, src_edge)
        # Dynamic Message Passing
        dmp = DynamicMsgPass(hidden_dim=self.hidden_dim)
        # h_v_: [N, hidden_dim]
        h_v_ = comp
        for _ in range(self.hops):
            h_v_ = dmp(h_v_, src_edge, E_c, ptr)
        # Hybrid Message Passing
        hmp = HybridMsgPass(hidden_dim=self.hidden_dim)
        f_v_k = hmp(comp, h_v, h_v_)
        # concatenate the state of BiLSTM with the graph encoding results
        # h: [batch_size, com_len, hidden_dim], f_v_k: [node_num, hidden_dim]
        graph_enc_outs = []
        # graph_reps: [batch_size, hidden_dim]
        graph_reps = global_max_pool(f_v_k, batch=batch)
        for graph_id in range(batch_size):
            start = ptr[graph_id]
            end = ptr[graph_id + 1]
            graph_rep = graph_reps[graph_id].unsqueeze(0)
            h_n = hn[graph_id].unsqueeze(0)
            z_ = torch.cat((graph_rep, h_n, graph_rep * h_n, graph_rep - h_n), dim=-1)
            z = self.Sigmoid(self.W_z(z_))
            dec_init = z * graph_rep + (1 - z) * h_n
            enc_out = torch.cat((h[graph_id], f_v_k[start: end]), dim=0)
            graph_enc_outs.append((enc_out, dec_init))
        # graph_enc_outs: ([batch_size, node_num + com_len, hidden_dim], [1, hidden_dim])
        print(graph_enc_outs[0][0].shape, graph_enc_outs[0][1].shape)
        return graph_enc_outs

class DecoderCellState(
        namedtuple('DecoderCellState',
                   ('context', 'state', 'alignments'), defaults=[None])):
    def batch_select(self, indices):
        select = lambda x, dim=0: x.index_select(dim, indices)
        return self._replace(context = select(self.context),
                             state = tuple_map(select, self.state, dim=1),
                             alignments = tuple_map(select, self.alignments))


class DecoderCell(nn.Module):
    def __init__(self, embed_dim, hidden_dim,  dropout, glob_attn, num_layers=1,
                 memory_dim=None, input_feed=True, use_attn_layer=True):
        super(DecoderCell, self).__init__()
        self.glob_attn = glob_attn
        self.input_feed = glob_attn and input_feed
        self.use_attn_layer = glob_attn and use_attn_layer
        self.dropout = nn.Dropout(dropout)

        if memory_dim is None:
            memory_dim = hidden_dim
        cell_in_dim, context_dim = embed_dim, memory_dim
        if glob_attn is not None:
            self.attention = GlobalAttention(hidden_dim, memory_dim, glob_attn)

            if use_attn_layer:
                self.attn_layer = nn.Linear(context_dim + hidden_dim, hidden_dim)
                context_dim = hidden_dim
            if input_feed:
                cell_in_dim += context_dim
        self.cell = nn.LSTM(cell_in_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)

    def forward(self, tgt_embed, prev_state, memory=None, src_lengths=None):
        #        perform one decoding step
        cell_input = self.dropout(tgt_embed)  # batch_size * embed_size

        if self.glob_attn is not None:
            if self.input_feed:
                cell_input = torch.cat([cell_input, prev_state.context], 1).unsqueeze(1)
                # batch_size * 1 * (embed_size + units)
            output, state = self.cell(cell_input, prev_state.state)
            output = output.squeeze(1)  # batch_size * units
            context, alignments = self.attention(output, memory, src_lengths)
            if self.use_attn_layer:
                context = torch.cat([context, output], 1)
                context = torch.tanh(self.attn_layer(context))
            context = self.dropout(context)
            return DecoderCellState(context, state, alignments)
        else:
            output, state = self.cell(cell_input, prev_state.state)
            output = output.squeeze(1)
            return DecoderCellState(output, state)

class Decoder(nn.Module):
    def __int__(self, field, embed_dim, hidden_dim, glob_attn, num_layers, dropout, **kwargs):
        super(Decoder, self).__int__()
        com_vocab_size = len(field.vocab)
        pad_id = field.vocab.stoi['<pad>']
        self.field = field
        self.glob_attn = glob_attn
        self.hidden_dim = hidden_dim

        self.ComEembedding = nn.Embedding(com_vocab_size, embed_dim, padding_idx=pad_id)
        self.OutLayer = nn.Linear(hidden_dim, com_vocab_size, bias=False)
        self.cell = DecoderCell(embed_dim, hidden_dim, num_layers, dropout,
                                glob_attn, **kwargs)

    @property
    def attn_history(self):
        if self.cell.hybrid_mode:
            return self.cell.attention.attn_history
        elif self.glob_attn:
            return 'std'

    def initialize(self, enc_final):
        init_context = torch.zeros(enc_final[0].shape[1], self.units,
                                   device=enc_final[0].device)
        return DecoderCellState(init_context, enc_final)

    def forward(self, trg_input, enc_final, memory=None, src_lengths=None, return_contex=False):
        # com_embed: [batch_size, com_len, hidden_dim], enc_final: [batch_size, node_num + com_len, hidden_dim]
        com_embed = self.ComEembedding(trg_input)

        output_seqs, attn_history = [], []
        for graph_enc in enc_final:
            # dec_input: [node_num + com_len, hidden_dim]
            dec_input = graph_enc[0]
            # dec_init: [1, hidden_dim]
            dec_init = graph_enc[1]

        return