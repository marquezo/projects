import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

#################################################################################################################
#
# The controller uses an LSTM or an MLP
#
#################################################################################################################
class controller(nn.Module):
    def __init__(self, ctrl_type, input_size, hidden_size, batch_size=1,
                 num_layers=1):
        super(controller, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctrl_type = ctrl_type
        self.batch_size = batch_size
        self.num_layers = num_layers

        if self.ctrl_type == "lstm":
            self.controller = nn.LSTM(input_size, hidden_size, num_layers)
            self.hidden0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.05)
            self.cell0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.05)
            self.init_parameters()
        elif self.ctrl_type == "ffnn":
            self.controller = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        if self.ctrl_type == "lstm":
            x, self.hidden = self.controller(x.view(1, self.batch_size, -1), self.hidden)
            x = x.view(self.batch_size, -1)
        elif self.ctrl_type == "ffnn":
            x = self.controller(x)
        return F.tanh(x)

    # Crucial for learning: the hidden and cell state of the LSTM at the start of each mini-batch must be learned
    def _init_hidden(self):
        hidden_state = self.hidden0.clone().repeat(1, self.batch_size, 1)
        cell_state = self.cell0.clone().repeat(1, self.batch_size, 1)
        if use_cuda:
            hidden_state, cell_state = hidden_state.cuda(), cell_state.cuda()
        return hidden_state, cell_state

    def init_parameters(self):
        for param in self.controller.parameters():
            if param.dim() == 1:
                nn.init.constant(param, 0)
            else:
                stdev = 5 / (np.sqrt(self.input_size + self.hidden_size))
                nn.init.uniform(param, -stdev, stdev)