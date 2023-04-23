from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMCell(nn.Module):
    def __init__(self, inp_size: int, hidden_size: int, seq_length: int = 6, seq_out: int = 1):
        super(LSTMCell, self).__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.seq_out = seq_out

        self.gates = nn.Linear(in_features=inp_size, out_features=4 * hidden_size)
        self.cell_to_out = nn.Linear(in_features=seq_length, out_features=seq_out)
        self.init_weights()

    # modeled after torch.nn.LSTM
    def forward(self, inp: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inp:
        :param hx: [output from previous cell, hidden cell]
        :return:
        """
        if hx[0] is None or hx[1] is None:
            hx = Variable(inp.new_zeros(inp.size(0), 1, self.inp_size))
            hx = (hx, hx)

        hx, cx = hx

        gates_inps = torch.cat([inp, hx], dim=1)
        gates = self.gates(gates_inps)

        # Get gates (i_t, f_t, g_t, o_t)
        inp_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, -1)

        # input gate
        # i_t = sigmoid(fcn(Concatenate[h_t_1, x_t]))
        i_t = torch.sigmoid(inp_gate)
        # forget gate
        # f_t = sigmoid(fcn(Concatenate[h_t_1, x_t]))
        f_t = torch.sigmoid(forget_gate)
        # cell remember gate
        # g_t = tanh(fcn(Concatenate[h_t_1, x_t]))
        g_t = torch.tanh(cell_gate)
        # output gate
        # o_t = sigmoid(fcn(Concatenate[h_t_1, x_t]))
        o_t = torch.sigmoid(out_gate)

        # cx_1 = cx * f_t + i_t * g_t
        cx_1 = torch.add(torch.mul(cx, f_t), torch.mul(i_t, g_t))

        # hx_1 = o_t * tanh(cx_1) -> mapped to output prediction length
        hx_1 = self.cell_to_out(torch.mul(o_t, torch.tanh(cx_1)).permute(0, 2, 1)).permute(0, 2, 1)

        return hx_1, cx_1

    def init_weights(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)


class LSTM(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        embedding_size: int,
        n_hidden_neurons: int,
        n_layers: int = 1,
        n_gram: int = 6,
    ):
        super(LSTM, self).__init__()
        self.n_tokens = n_tokens
        self.embedding_size = embedding_size
        self.n_hidden_neurons = n_hidden_neurons
        self.n_layers = n_layers
        self.n_gram = n_gram

        # embed high dimensional vocab size into lower dimensional embedding
        # helps reduce number of parameters and better represent incoming information
        self.encoder = nn.Embedding(n_tokens, embedding_size)
        self.lstm_cell = LSTMCell(embedding_size, n_hidden_neurons)
        # remap output from lstm to vocab size higher dimension
        self.decoder = nn.Linear(n_hidden_neurons, n_tokens)

    def forward(self, x_t: torch.Tensor, h_t_1: torch.Tensor, C_t_1: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        emb_x_t = self.encoder(x_t)
        hx = self.lstm_cell(emb_x_t, (h_t_1, C_t_1))
        output = hx[0]  # only have one hidden layer, so output equals hx

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hx[0], hx[1]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
