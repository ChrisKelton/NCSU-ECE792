from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMCell(nn.Module):
    def __init__(self, inp_size: int, hidden_size: int, n_grams: int = 6):
        super(LSTMCell, self).__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.n_grams = n_grams

        self.gates = nn.Linear(in_features=inp_size, out_features=4 * hidden_size)
        self.cell_to_out = nn.Linear(in_features=n_grams, out_features=1)
        # self.ht1 = nn.Linear(in_features=hidden_size, out_features=4 * hidden_size)
        self.reset_weights()

    def reset_weights(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, inp: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if hx[0] is None or hx[1] is None:
            hx = Variable(inp.new_zeros(inp.size(0), 1, self.inp_size))
            hx = (hx, hx)

        hx, cx = hx

        gates_inps = torch.cat([inp, hx], dim=1)
        gates = self.gates(gates_inps)

        # Get gates (i_t, f_t, g_t, o_t)
        inp_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, -1)

        i_t = torch.sigmoid(inp_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(out_gate)

        cy = torch.add(torch.mul(cx, f_t), torch.mul(i_t, g_t))

        hy = self.cell_to_out(torch.mul(o_t, torch.tanh(cy)).permute(0, 2, 1)).permute(0, 2, 1)

        return hy, cy


class LSTM(nn.Module):
    # C_t_1: torch.Tensor

    def __init__(self, n_tokens: int, embedding_size: int, n_hidden_neurons: int, n_layers: int = 1, n_gram: int = 6, tie_weights: bool = False):
        super(LSTM, self).__init__()
        self.n_tokens = n_tokens
        self.embedding_size = embedding_size
        self.n_hidden_neurons = n_hidden_neurons
        self.n_layers = n_layers
        self.n_gram = n_gram
        self.tie_weights = tie_weights

        self.encoder = nn.Embedding(n_tokens, embedding_size)
        self.lstm_cell = LSTMCell(embedding_size, n_hidden_neurons)
        # # multiply in_feats by 2, b/c we concatenate x_t & h_t_1
        # self.forget_gate_layer = nn.Linear(in_features=embedding_size, out_features=n_hidden_neurons)
        # self.C_t_1 = torch.zeros((n_layers, n_hidden_neurons))
        #
        # self.input_gate_layer = nn.Linear(in_features=embedding_size, out_features=n_hidden_neurons)
        # self.tanh_candidate_layer = nn.Linear(in_features=embedding_size, out_features=n_hidden_neurons)
        #
        # self.output_update_layer = nn.Linear(in_features=embedding_size, out_features=n_hidden_neurons)
        #
        # self.sigmoid_act_func = nn.Sigmoid()
        # self.tanh_act_func = nn.Tanh()

        self.decoder = nn.Linear(n_hidden_neurons, n_tokens)

    def forward(self, x_t: torch.Tensor, h_t_1: torch.Tensor, C_t_1: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        emb_x_t = self.encoder(x_t)
        hx = self.lstm_cell(emb_x_t, (h_t_1, C_t_1))
        output = hx[0]
        # gate_inputs = torch.cat([h_t_1, emb_x_t], dim=-1)
        # forget_gate_output = self.sigmoid_act_func(self.forget_gate_layer(gate_inputs))
        # C_t_1 = torch.mul(C_t_1, forget_gate_output)
        #
        # input_gate_output = self.sigmoid_act_func(self.input_gate_layer(gate_inputs))
        # tanh_candidate_output = self.tanh_act_func(self.tanh_candidate_layer(gate_inputs))
        # candidate_state_update = torch.mul(tanh_candidate_output, input_gate_output)
        # C_t_1 = torch.add(C_t_1, candidate_state_update)
        #
        # output_update = self.sigmoid_act_func(self.output_update_layer(gate_inputs))
        # tanh_new_candidate_state = torch.tanh(C_t_1)
        #
        # h_t_1 = torch.mul(tanh_new_candidate_state, output_update)
        # output = torch.clone(h_t_1)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hx[0], hx[1]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        weight = next(self.parameters()).data
        return Variable(weight.new(batch_size, self.n_layers, self.n_hidden_neurons).zero_())