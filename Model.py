import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, encoder, decoder, out_dim, device):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.out_dim = out_dim

    def forward(self, src, rev, sim_score, trg, teacher_forcing=True):
        # trg: [batch_size, trg_len]
        batch_size = len(src.ptr) - 1
        trg_len = trg.shape[1]
        trg_vocab_size = self.out_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        enc_outputs = self.encoder(src, rev, sim_score)
        # dec_input: [batch_size, 1]
        dec_input = trg[0, 0].view(1, 1)

        for graph_id in range(batch_size):
            # enc_o: [node_num + com_len, hidden_dim]
            enc_o = enc_outputs[graph_id][0]
            # s_: [hidden_dim]
            s_ = enc_outputs[graph_id][1]
            for t in range(1, trg_len):
                # dec_out: [1, out_dim], s_: [dec_hidden_dim]
                dec_out, s_ = self.decoder(dec_input, (enc_o, s_))
                outputs[graph_id][t] = dec_out.squeeze(0)
                top1 = dec_out.argmax(1)
                dec_input = trg[graph_id][t].view(1, 1) if teacher_forcing else top1.view(1, 1)
        return outputs
