import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from NTM import NTM

use_cuda = torch.cuda.is_available()


# Sequence, batch, input
def generate_input_example(sequence_length=None, batch_size=1):
    # length of binary vectors fed to models
    vector_length = 8

    # length of sequence of binary vectors
    if sequence_length is None:
        # generate random sequence length between 1 and 20
        sequence_length = np.random.randint(1, 21)

    data = np.random.randint(2, size=(sequence_length, batch_size, vector_length + 1))

    # making sure all data has no EOS (no 1 at 9th position)
    data[:, :, -1] = 0.0

    padding = np.zeros((sequence_length, batch_size, vector_length + 1))

    delimiter = np.zeros((1, batch_size, vector_length + 1))
    delimiter[:, :, -1] = 1.0

    inputs = np.concatenate((data, delimiter, padding))

    delimiter = np.zeros((1, batch_size, vector_length + 1))
    targets = np.concatenate((padding, delimiter, data))

    # convert to torch tensors
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).float()

    return inputs, targets


def show_last_example(inputs, outputs, targets):
    inputs = inputs[:, 0].data.cpu().numpy()
    outputs = outputs[:, 0].data.cpu().numpy()
    targets = targets[:, 0].data.cpu().numpy()

    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.matshow(inputs.T, aspect='auto')
    ax2.matshow(targets.T, aspect='auto')
    ax3.matshow(outputs.T, aspect='auto')

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)

    plt.show()
    plt.clf()


def train(model, n_updates=100000, learning_rate=1e-4, print_every=100,
          show_plot=False, sequence_length=None):
    if use_cuda:
        model = model.cuda()

    criterion = nn.BCELoss()
    # original paper uses RMSProp with momentum 0.9
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9, alpha=0.95)

    loss_tracker = []
    cost_per_seq = 0

    for update in range(n_updates):

        optimizer.zero_grad()

        inputs, targets = generate_input_example(batch_size=model.batch_size, sequence_length=sequence_length)
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)

        output_len = outputs.shape[0] // 2

        loss = criterion(outputs[-output_len:], targets[-output_len:])
        cost_per_seq += loss.data[0]

        loss.backward()
        parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in parameters:
            p.grad.data.clamp_(-10, 10)
        optimizer.step()

        if update % print_every == 0:

            if update != 0:
                cost_per_seq /= print_every
            loss_tracker.append(cost_per_seq)
            print("Number of sequences processed : %d ----- Cost per sequence(bits) : %.6f" % (
            update * model.batch_size, loss_tracker[-1]))

            if show_plot:
                show_last_example(inputs, outputs, targets)

            cost_per_seq = 0
    return loss_tracker

if __name__ == '__main__':
    lstm_ntm = NTM('lstm', batch_size=1)

    #state = torch.load('model_stable.pt')
    state = torch.load('/home/orlandom/Documents/UdeM/H2018/IFT6135/A3/model_working.pt')
    lstm_ntm.load_state_dict(state)

    print("Number of parameters for LSTM-NTM : %d" % lstm_ntm.number_of_parameters())
    lstm_ntm_loss = train(lstm_ntm, learning_rate=1e-4, n_updates=1, print_every=1, show_plot=True, sequence_length=20)
    #torch.save(lstm_ntm.state_dict(), 'model_stable.pt')
