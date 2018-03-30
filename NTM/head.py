import torch
from torch import nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

##################################################################################################
#
# Each head has 5 linear layers:
#   For the key parameter
#   For the beta parameter
#   For the blending parameter
#   For the shift parameter
#   For the gamma parameter
# It also has a refernce to the memory and the previous weight since it needs them to
# calculate the addressing
# The output of its forward() method is a normalized new weight
#
##################################################################################################
class head(nn.Module):
    def __init__(self, hidden_size, weight_size, key_size, shift_range=1, batch_size=1):
        super(head, self).__init__()

        self.hidden_size = hidden_size
        self.weight_size = weight_size
        self.key_size = key_size
        self.shift_range = shift_range
        self.batch_size = batch_size

        self.fc_key = nn.Linear(hidden_size, key_size)
        self.fc_beta = nn.Linear(hidden_size, 1)
        self.fc_blending = nn.Linear(hidden_size, 1)
        self.fc_shift = nn.Linear(hidden_size, 2 * shift_range + 1)
        self.fc_gamma = nn.Linear(hidden_size, 1)

        self.init_parameters()

    def init_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_normal(self.fc_key.weight)
        nn.init.constant(self.fc_key.bias, 0)

        nn.init.xavier_uniform(self.fc_beta.weight)
        nn.init.constant(self.fc_beta.bias, 0)

        nn.init.xavier_uniform(self.fc_blending.weight)
        nn.init.constant(self.fc_blending.bias, 0)

        nn.init.xavier_uniform(self.fc_shift.weight)
        nn.init.constant(self.fc_shift.bias, 0)

        nn.init.xavier_normal(self.fc_gamma.weight)
        nn.init.constant(self.fc_gamma.bias, 0)

    def forward(self, x, memory, prev_weight):
        self.memory = memory
        self.prev_weight = prev_weight
        self.key = F.relu(self.fc_key(x))
        self.beta = F.softplus(self.fc_beta(x))
        self.blending = F.sigmoid(self.fc_blending(x))
        self.shift = F.softmax(self.fc_shift(x), 1)
        self.gamma = F.relu(self.fc_gamma(x)) + 1

        self._addressing()

        return self.weight

    def _addressing(self):
        self._content_addressing()
        self._interpolation()
        self._convolutional_shift()
        self._sharpening()

    def _content_addressing(self):
        self.weight = F.softmax(self.beta * F.cosine_similarity(self.key.unsqueeze(1), self.memory, dim=-1), 1)

    def _interpolation(self):
        self.weight = self.blending * self.weight + (1 - self.blending) * self.prev_weight

    def _convolutional_shift(self):
        tmp = torch.zeros_like(self.weight)
        # expanding weight vector for same convolution
        self.weight = torch.cat((self.weight[:, -1:], self.weight, self.weight[:, :1]), dim=1)
        for b in range(self.batch_size):
            tmp[b] = F.conv1d(self.weight[b].view(1, 1, -1), self.shift[b].view(1, 1, -1))
        self.weight = tmp

    def _sharpening(self):
        self.weight = self.weight ** self.gamma
        self.weight = torch.div(self.weight, torch.sum(self.weight, dim=1).unsqueeze(1))