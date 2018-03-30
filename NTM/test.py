
from NTM import NTM
from main import generate_input_example, show_last_example
from torch.autograd import Variable
import torch

use_cuda = torch.cuda.is_available()

def eval(model, example, show_plot=True):
    if use_cuda:
        model = model.cuda()

    inputs = Variable(example)
    output = model(inputs)

    if show_plot:
        show_last_example(inputs, output, output)

lstm_ntm = NTM('lstm', batch_size=1)

state = torch.load('/home/orlandom/Documents/UdeM/H2018/IFT6135/A3/model_mem.pt')
lstm_ntm.load_state_dict(state)

example, _ = generate_input_example(batch_size=1, sequence_length=10)

#example = example[:5]

eval(lstm_ntm, example)