from CombinatorialRL import CombinatorialRL
from torch.utils.data import DataLoader
from inference import *
import torch
import pickle

embedding_size = 128
hidden_size    = 128
n_glimpses = 1
tanh_exploration = 10
use_tanh = True
seq_len=20

max_grad_norm = 1.
USE_CUDA = torch.cuda.is_available()

tsp20 = CombinatorialRL(
        embedding_size,
        hidden_size,
        seq_len,
        n_glimpses,
        tanh_exploration,
        use_tanh,
        attention="Bahdanau",
        use_cuda=USE_CUDA)

state = torch.load('Model/TSP20/tsp_model_TSP20_001.pt', map_location=lambda storage, loc:storage)
tsp20.load_state_dict(state)

if USE_CUDA:
        tsp20 = tsp20.cuda()

tsp20_dataset = pickle.load(open("test_data_set_20", "rb"))

tsp20_loader = DataLoader(tsp20_dataset, batch_size=128, shuffle=False, num_workers=1)

tour_lengths = np.zeros(1)

for batch_id, sample_batch in enumerate(tsp20_loader):

        inputs = Variable(sample_batch)

        if USE_CUDA:
                inputs = inputs.cuda()

        R, probs, actions, action_idxs, critic_evals = tsp20(inputs, T=2.0)

        if (tour_lengths == np.zeros(1)).all():
                tour_lengths = R.data.numpy()
        else:
                tour_lengths = np.append(tour_lengths, R.data.numpy())

print (tour_lengths.mean())

file = open('results_greedy_tsp20', 'wb')
pickle.dump(tour_lengths, file)
file.close()