from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, batch_size: int):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        # multiply in_feats by 2, b/c we concatenate x_t & h_t_1
        self.forget_gate_layer = nn.Linear(in_features=in_feats + out_feats, out_features=out_feats)
        self.h_t_1 = torch.zeros((batch_size, 1))
        self.C_t_1 = torch.zeros((batch_size, out_feats))

        self.input_gate_layer = nn.Linear(in_features=in_feats + out_feats, out_features=out_feats)
        self.tanh_candidate_layer = nn.Linear(in_features=in_feats + out_feats, out_features=out_feats)

        self.output_update_layer = nn.Linear(in_features=in_feats + out_feats, out_features=out_feats)

        self.sigmoid_act_func = nn.Sigmoid()
        self.tanh_act_func = nn.Tanh()

    def forward(self, x_t: torch.Tensor, h_t_1: torch.Tensor, C_t_1: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        gate_inputs = torch.cat([h_t_1, x_t], dim=1)
        forget_gate_output = self.sigmoid_act_func(self.forget_gate_layer(gate_inputs))
        C_t_1 = torch.mul(C_t_1, forget_gate_output)

        input_gate_output = self.sigmoid_act_func(self.input_gate_layer(gate_inputs))
        tanh_candidate_output = self.tanh_act_func(self.tanh_candidate_layer(gate_inputs))
        candidate_state_update = torch.mul(tanh_candidate_output, input_gate_output)
        C_t_1 = torch.add(C_t_1, candidate_state_update)

        output_update = self.sigmoid_act_func(self.output_update_layer(gate_inputs))
        tanh_new_candidate_state = torch.tanh(C_t_1)

        h_t_1 = torch.mul(tanh_new_candidate_state, output_update)

        return h_t_1, C_t_1

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        weight = next(self.parameters()).data
        return Variable(weight.new(batch_size, self.out_feats).zero_())