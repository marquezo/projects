import torch
from torch import nn
from controller import controller
from head import head
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class NTM(nn.Module):
    def __init__(self, ctrl_type, input_size=9, hidden_size=100, num_layers=1,
                 n_mem_loc=128, mem_loc_len=20, shift_range=1, batch_size=1):
        super(NTM, self).__init__()

        self.ctrl_type = ctrl_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = (n_mem_loc, mem_loc_len)
        self.shift_range = shift_range
        self.batch_size = batch_size

        if ctrl_type not in ['lstm', 'ffnn']:
            raise Exception("Controller type '%s' not supported. "
                            "Please choose between 'lstm' and 'ffnn'." % ctrl_type)

        # creating controller, read head and write head
        self.controller = controller(ctrl_type, input_size + mem_loc_len, hidden_size, batch_size=batch_size)
        self.write_head = head(hidden_size, n_mem_loc, mem_loc_len, batch_size=batch_size)
        self.read_head = head(hidden_size, n_mem_loc, mem_loc_len, batch_size=batch_size)

        self.fc_erase = nn.Linear(hidden_size, mem_loc_len)
        self.fc_add = nn.Linear(hidden_size, mem_loc_len)
        self.fc_out = nn.Linear(mem_loc_len, input_size)

        self.memory0 = nn.Parameter(torch.randn(1, self.memory_size[0],
                                                self.memory_size[1]) * 0.05)
        self.write_weight0 = nn.Parameter(torch.randn(1, self.memory_size[0]) * 0.05)
        self.read_weight0 = nn.Parameter(torch.randn(1, self.memory_size[0]) * 0.05)

        self.read0 = nn.Parameter(torch.randn(1, self.memory_size[1]) * 0.05)

        self.init_parameters()

    def init_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform(self.fc_erase.weight)
        nn.init.constant(self.fc_erase.bias, 0)

        nn.init.xavier_uniform(self.fc_add.weight)
        nn.init.normal(self.fc_add.bias, 0)

        nn.init.xavier_uniform(self.fc_out.weight)
        nn.init.constant(self.fc_out.bias, 0)

    def forward(self, x):

        self.ntm_out = None

        self.memory = self._init_memory()
        self.prev_write_weight, self.prev_read_weight = self._init_weight()
        self.read = self._init_read()

        if self.ctrl_type == "lstm":
            self.controller.hidden = self.controller._init_hidden()

        #Take a slice of length batch_size x input (original size of input + M)
        for input in x:

            input = torch.cat((input, self.read), dim=1)
            ht = self.controller(input)

            self.write_weight = self.write_head(ht, self.memory, self.prev_write_weight)
            self.read_weight = self.read_head(ht, self.memory, self.prev_read_weight)

            self.prev_write_weight = self.write_weight
            self.prev_read_weight = self.read_weight


            self.erase = F.sigmoid(self.fc_erase(ht))
            self.add = F.sigmoid(self.fc_add(ht))

            self._write()
            self._read()

            out = self.fc_out(self.read).unsqueeze(0)
            out = F.sigmoid(out)

            if self.ntm_out is None:
                self.ntm_out = out
            else:
                self.ntm_out = torch.cat((self.ntm_out, out))

        return self.ntm_out

    def _read(self):
        self.read = torch.matmul(self.read_weight.unsqueeze(1), self.memory).view(self.batch_size, -1)

    def _write(self):
        erase_tensor = torch.matmul(self.write_weight.unsqueeze(-1), self.erase.unsqueeze(1))
        add_tensor = torch.matmul(self.write_weight.unsqueeze(-1), self.add.unsqueeze(1))
        self.memory = self.memory * (1 - erase_tensor) + add_tensor

    def _init_memory(self):
        memory = self.memory0.clone().repeat(self.batch_size, 1, 1)
        if use_cuda:
            memory = memory.cuda()
        return memory

    def _init_weight(self):
        read_weight = self.read_weight0.clone().repeat(self.batch_size, 1)
        write_weight = self.write_weight0.clone().repeat(self.batch_size, 1)
        if use_cuda:
            read_weight, write_weight = read_weight.cuda(), write_weight.cuda()


        #print torch.sum(read_weight)

        read_weight = F.softmax(read_weight, 1)
        write_weight = F.softmax(write_weight, 1)

        return write_weight, read_weight

    def _init_read(self):
        readvec = self.read0.clone().repeat(self.batch_size, 1)

        if use_cuda:
            readvec = readvec.cuda()
        return readvec

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)