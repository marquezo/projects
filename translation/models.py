import torch.nn as nn
import torch
import torch.nn.functional as F
from attention_layer import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=self.num_layers)

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
    def __init__(self, output_size, hidden_size, num_layers, dropout_p=0.1, use_attention=False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=self.num_layers)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.use_attention = use_attention

        if self.use_attention:
            self.attention = Attention(self.hidden_size).to(device)

    def forward(self, input, hidden, encoder_outputs=None):

        output = self.embedding(input)

        # We are doing inference
        if output.dim() == 1:
            output = output.view(1, 1, -1)

        output = self.dropout(output)

        # Here is where we can use attention: output is the query and encoder_outputs are the ref
        # The idea is that instead of conditioning on one vector containing all the input,
        # condition on specific parts of the input
        if self.use_attention and encoder_outputs is not None:
            output = self.attention(output.squeeze(0), encoder_outputs)
            #print("using attention in decoder:", output.size())

        output = F.relu(output) # [ batch_size x seq_len x hidden_size ]
        print(output.size(), hidden.size())
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
