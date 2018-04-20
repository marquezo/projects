import torch
import torch.nn as nn

import torch.nn.functional as F
from Attention import Attention


class Critic(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 num_processing,
                 n_glimpses,
                 use_cuda=False):
        super(Critic, self).__init__()

        # self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len
        self.num_processing = num_processing
        self.use_cuda = use_cuda

        # self.embedding = GraphEmbedding(2, embedding_size, use_cuda=use_cuda)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        # self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.glimpse = Attention(hidden_size, use_tanh=True, name='Bahdanau', use_cuda=use_cuda)

        self.fc1.weight = torch.nn.init.uniform(self.fc1.weight, -0.08, 0.08)
        self.fc2.weight = torch.nn.init.uniform(self.fc2.weight, -0.08, 0.08)

        for p in self.encoder.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                nn.init.uniform(p, -0.08, 0.08)

    def forward(self, inputs, input_embedded):
        """
        Args:
            inputs: [batch_size x 1 x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        assert seq_len == self.seq_len

        # embedded = self.embedding(inputs)

        # The encoder simply runs the embedding
        encoder_outputs, (hidden, context) = self.encoder(input_embedded)
        """encoder_outputs: [batch_size x seq_len x hidden_size]"""

        # The first input to the decoder is the last hidden state
        # decoder_input = torch.t(hidden) #Batch size has to be the first dimension, so swap first and second dimensions

        # Init decoder's hidden and reuse the encoder's context
        # hidden = Variable(torch.zeros(1, batch_size, hidden_size))
        # context = Variable(torch.zeros(context.size()))

        #         print (decoder_input.size())

        query = torch.t(hidden).squeeze()

        # For each step in the sequence
        for i in range(self.num_processing):
            # _, (hidden, context) = self.decoder(decoder_input, (hidden, context))

            # query = hidden.squeeze(0) #[hidden_size x 1] (or the other way around)

            # Do the glimpse
            ref, logits = self.glimpse(query, encoder_outputs)
            # logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)

            # [batch_size x hidden_size]
            decoder_input = query  # .unsqueeze(1)

        # Do fully connected part   TODO: batch norm
        output = self.fc1(query)
        output = F.relu(output)
        output = self.fc2(output)

        # list of seq_len containing[batch_size x seq_len], list of seq_len containing [batch_size]
        return output