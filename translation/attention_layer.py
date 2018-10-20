import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size)
        # use a 1D convolution instead of a linear layer to not depend on sequence length
        self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)
        V = torch.FloatTensor(hidden_size, device=device)

        self.V = nn.Parameter(V)
        self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   [batch_size x seq_len x hidden_size]
        """
        #print("in attention", query.size(), ref.size())
        batch_size = ref.size(0)
        seq_len = ref.size(1)

        ref = ref.permute(0, 2, 1) # [batch_size x hidden_size x seq_len]
        query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
        ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
        expanded_query = query.repeat(1, 1, seq_len)  # [batch_size x hidden_size x seq_len]
        V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size x 1 x hidden_size]
        logits = torch.bmm(V, torch.tanh(expanded_query + ref)).squeeze(1)
        weights = F.softmax(logits, dim=1)

        # Compute the weighted sum of annotations: [batch_size x hidden_size x seq_len] x [batch_size x seq_len x 1]
        expected_annotation = torch.bmm(ref, weights.unsqueeze(2))

        return expected_annotation

