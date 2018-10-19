import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=self.num_layers)

        for p in self.gru.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                nn.init.uniform_(p, -0.08, 0.08)

    def forward(self, input, hidden):
        """
        :param input: dimensions batch_size x sequence_length
        :param hidden: 1 x batch_size x hidden layer dimensions
        :return:
        """
        embedded = self.embedding(input)

        output = embedded
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=self.num_layers)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.out.weight = nn.init.xavier_uniform_(self.out.weight)

        for p in self.gru.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                nn.init.uniform_(p, -0.08, 0.08)

    def forward(self, input, hidden):

        output = self.embedding(input)

        # We are doing inference
        if output.dim() == 1:
            output = output.view(1, 1, -1)

        output = self.dropout(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
