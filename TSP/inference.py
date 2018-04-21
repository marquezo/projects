
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

def reward_single_input(sample_solution):
    """
    Args:
        sample_solution list of length 'seq_len' of [batch_size x input_size (2 since doing 2D coordinates)]
    """
    n = sample_solution.size(0)
    tour_len = 0

    for i in range(n - 1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i + 1])

    tour_len += torch.norm(sample_solution[n - 1] - sample_solution[0])

    return tour_len


def shuffle_tensor(tensor):
    shuffle_indexes = torch.randperm(tensor.size(-1))
    tensor_shuffled = torch.FloatTensor(tensor.t().size())

    for i in range(tensor.size(-1)):
        tensor_shuffled[i] = tensor.t()[shuffle_indexes[i]]

    return tensor_shuffled.t()


def create_graph(num_nodes):
    x = torch.FloatTensor(2, num_nodes).uniform_(0, 1)
    return x


#################################################################################
# input: tensor of shape 2 x seq_len
# model: object of type Combinatorial RL
# num_candidates: how many solutions to try
# batch_size
# alpha: for the exponential moving average
# lr: learning rate to use
#################################################################################
def active_search(input, model, num_candidates, batch_size, alpha=0.99, lr=1e-6):
    baseline = torch.zeros(1)
    actor_optim = optim.Adam(model.actor.parameters(), lr=lr)

    # Create random solution
    soln = shuffle_tensor(input)
    soln_tour_length = reward_single_input(soln.t())
    n = torch.ceil(torch.FloatTensor([num_candidates / batch_size]))

    for batch_id in range(n):

        shuffled_input = input.unsqueeze(0).repeat(batch_size, 1, 1)

        # Shuffled the input for batch_size times
        for i in range(batch_size):
            shuffled_input[i] = shuffle_tensor(shuffled_input[i])

        shuffled_input = Variable(shuffled_input)
        R, probs, actions, action_idxs, critic_evals = model(shuffled_input)

        # R is tensor of size batch_size
        # Pick the shortest tour
        idx_min_tour = np.argmin(R.data)
        min_tour_length = R[idx_min_tour]

        # print (soln_tour_length)

        if (min_tour_length.data[0] < soln_tour_length):
            soln_tour_length = min_tour_length.data[0]
            soln = torch.zeros((input.size(-1), 2))

            # for seq len, get the new solution
            for i in range(input.size(-1)):
                soln[i] = actions[i][idx_min_tour].data

        if batch_id == 0:
            baseline = R.mean()
        else:
            baseline = (baseline * alpha) + ((1. - alpha) * R.mean())

        advantage = R - baseline

        logprobs = 0

        for prob in probs:
            logprob = torch.log(prob)
            logprobs += logprob

        if logprobs.data[0] < -1000:
            logprobs = Variable(torch.FloatTensor([0.]), requires_grad=True)

        reinforce = advantage * logprobs
        actor_loss = reinforce.mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(model.actor.parameters(),
                                      1.0, norm_type=2)

        actor_optim.step()

    return soln


#################################################################################
# input: tensor of shape 2 x seq_len
# model: object of type Combinatorial RL
# num_candidates: how many solutions to try
# batch_size
#################################################################################
def sample_solution(input, model, batch_size, T=1.0):
    shuffled_input = input.unsqueeze(0).repeat(batch_size, 1, 1)

    # Shuffled the input for batch_size times
    for i in range(batch_size):
        shuffled_input[i] = shuffle_tensor(shuffled_input[i])

    shuffled_input = Variable(shuffled_input)
    R, probs, actions, action_idxs, critic_evals = model(shuffled_input, T=T)

    # R is tensor of size batch_size
    # Pick the shortest tour
    idx_min_tour = np.argmin(R.data)
    min_tour_length = R[idx_min_tour]

    soln = torch.zeros((input.size(-1), 2))

    # for seq len, get the new solution
    for i in range(input.size(-1)):
        soln[i] = actions[i][idx_min_tour].data

    return soln, min_tour_length