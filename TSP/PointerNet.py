import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphEmbedding import GraphEmbedding
from Attention import Attention


class PointerNet(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 attention,
                 use_cuda=False):
        super(PointerNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len
        self.use_cuda = use_cuda

        self.embedding = GraphEmbedding(2, embedding_size, use_cuda=use_cuda)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention, use_cuda=use_cuda)
        self.glimpse = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention, use_cuda=use_cuda)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

        for p in self.encoder.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                nn.init.uniform(p, -0.08, 0.08)

        for p in self.decoder.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                nn.init.uniform(p, -0.08, 0.08)

    """
    idxs: indeces that were previously chosen
    logits: probabilities for current step
    """

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def forward(self, inputs, T=1.0):
        """
        Args:
            inputs: [batch_size x 1 x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        assert seq_len == self.seq_len

        embedded = self.embedding(inputs)

        # The encoder simply runs the embedding
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prev_probs = []
        prev_idxs = []
        mask = torch.zeros(batch_size, seq_len).byte()
        if self.use_cuda:
            mask = mask.cuda()

        idxs = None

        # The first input to the decoder is learned, as in the paper
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        # For each step in the sequence
        for i in range(seq_len):

            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)  # [hidden_size x 1] (or the other way around)

            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)

            # [batch size x seq_len]
            probs = F.softmax(logits / T, dim=1)

            # Give me the index that will be chosen: [batch_size]
            idxs = probs.multinomial().squeeze(1)

            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    print (seq_len)
                    print(' RESAMPLE!')
                    idxs = probs.multinomial().squeeze(1)
                    break

            # [batch_size x hidden_size]
            decoder_input = embedded[range(batch_size), idxs.data, :]

            prev_probs.append(probs)
            prev_idxs.append(idxs)

        # list of seq_len containing[batch_size x seq_len], list of seq_len containing [batch_size]
        return prev_probs, prev_idxs, embedded