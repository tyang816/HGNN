import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GatedGraphConv, GATConv, global_max_pool
from torch_geometric.utils import add_self_loops, softmax
from src.Utils import sequence_mask, tuple_map
import math
from collections import namedtuple


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
            # because the edge is directionless, each `key` of a node has different hidden value
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
        self.BiLSTM = nn.LSTM(hidden_dim, 128, bidirectional=True, batch_first=True)
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
            graph_enc_outs.append((enc_out, dec_init.squeeze(0)))
        # graph_enc_outs: ([node_num + com_len, hidden_dim], [hidden_dim])
        print(len(graph_enc_outs), graph_enc_outs[0][0].shape, graph_enc_outs[0][1].shape)
        return graph_enc_outs



class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s: [batch_size, dec_hid_dim]
        # enc_output: [batch_size, src_len, enc_hid_dim]
        batch_size = enc_output.shape[0]
        src_len = enc_output.shape[1]
        # repeat decoder hidden state src_len times
        # s: [batch_size, src_len, dec_hid_dim]
        # enc_output: [batch_size, src_len, enc_hid_dim]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        # energy: [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
        # attention: [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, com_field, embed_dim, hidden_dim, attention, dropout):
        super(Decoder, self).__init__()
        com_vocab_size = len(com_field.vocab)
        pad_id = com_field.vocab.stoi['<pad>']
        self.field = com_field
        self.attention = attention
        self.hidden_dim = hidden_dim

        self.ComEembedding = nn.Embedding(com_vocab_size, embed_dim, padding_idx=pad_id)
        self.OutLayer = nn.Linear(embed_dim + 2 * hidden_dim, com_vocab_size)
        self.LSTM = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.Dropout = nn.Dropout(dropout)


    def forward(self, trg_input, enc_final):
        # batch_size = 1, trg_len = 1
        # com_embed: [batch_size, trg_len, embed_dim]
        com_embed = self.ComEembedding(trg_input)
        # com_embed: [batch_size, trg_len, embed_dim]
        com_embed = self.Dropout(com_embed)

        # dec_input: [batch_size, node_num + com_len, hidden_dim]
        # print(f'enc {enc_final[0].shape}')
        enc_output = enc_final[0].unsqueeze(0)
        # dec_init: [batch_size, hidden_dim]
        dec_init = enc_final[1].unsqueeze(0)
        # a: [batch_size, 1, node_num + com_len]
        a = self.attention(dec_init, enc_output).unsqueeze(1)
        # c: [batch_size, 1, hidden_dim]
        c = torch.bmm(a, enc_output)
        lstm_input = torch.cat((com_embed, c), dim=-1)
        # dec_out: [batch_size, src_len(=1), dec_hid_dim]
        # dec_hidden: [batch_size, n_layers * num_directions, dec_hid_dim]
        dec_out, (hn, cn) = self.LSTM(lstm_input, (dec_init.unsqueeze(0), dec_init.unsqueeze(0)))
        # embedded: [batch_size, emb_dim]
        embedded = com_embed.squeeze(1)
        # dec_output: [batch_size, dec_hid_dim]
        dec_output = dec_out.squeeze(1)
        c = c.squeeze(1)
        # pred: [batch_size, output_dim]
        pred = self.OutLayer(torch.cat((dec_output, c, embedded), dim=1))
        return pred, hn.view(-1)