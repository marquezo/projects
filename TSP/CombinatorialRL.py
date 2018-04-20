
import torch
import torch.nn as nn
from PointerNet import PointerNet
from Critic import Critic
from torch.autograd import Variable

class CombinatorialRL(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 attention,
                 use_cuda=False):
        super(CombinatorialRL, self).__init__()
        self.use_cuda = use_cuda

        self.actor = PointerNet(
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            attention,
            use_cuda)

        self.critic = Critic(embedding_size=embedding_size,
                             hidden_size=hidden_size,
                             seq_len=seq_len,
                             num_processing=3,
                             n_glimpses=n_glimpses,
                             use_cuda=use_cuda)

    # Returns a vector of size equal to the mini-batch size
    def reward(self, sample_solution, USE_CUDA=False):
        """
        Args:
            sample_solution list of length 'seq_len' of [batch_size x input_size (2 since doing 2D coordinates)]
        """
        batch_size = sample_solution[0].size(0)
        n = len(sample_solution)

        tour_len = Variable(torch.zeros([batch_size]))

        if USE_CUDA:
            tour_len = tour_len.cuda()

        for i in range(n - 1):
            tour_len += torch.norm(sample_solution[i] - sample_solution[i + 1], dim=1)

        tour_len += torch.norm(sample_solution[n - 1] - sample_solution[0], dim=1)

        return tour_len

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        batch_size = inputs.size(0)
        input_size = inputs.size(1)  # 2 because we are doing 2D coordinates
        seq_len    = inputs.size(2)

        # list of seq_len containing[batch_size x seq_len], list of seq_len containing [batch_size]
        probs, action_idxs, input_embedded = self.actor(inputs)

        critic_evals = self.critic(inputs, input_embedded)

        #         print ("Combinatorial RL (inputs)")
        #         print (inputs)
        # print ("Combinatorial RL (action_idxs.size): ", action_idxs)

        actions = []

        """
        Transpose the inputs to have [batch_size, seq_len, input_size]
        """
        inputs = inputs.transpose(1, 2)

        # List of size seq_len
        for action_id in action_idxs:
            actions.append(inputs[range(batch_size), action_id.data, :])

        # actions now has the coordinates in the solution
        action_probs = []
        # List of size seq_len
        for prob, action_id in zip(probs, action_idxs):
            # We want to know the probability of taking each action (picking city) in the solution
            action_probs.append(prob[range(batch_size), action_id.data])

        # R is [batch_size x 1]
        R = self.reward(actions, self.use_cuda)

        return R, action_probs, actions, action_idxs, critic_evals