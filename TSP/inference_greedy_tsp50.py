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
seq_len=50

max_grad_norm = 1.
USE_CUDA = torch.cuda.is_available()

tsp50 = CombinatorialRL(
        embedding_size,
        hidden_size,
        seq_len,
        n_glimpses,
        tanh_exploration,
        use_tanh,
        attention="Bahdanau",
        use_cuda=USE_CUDA)

state = torch.load('Model/TSP50/tsp_model_T50_001.pt', map_location=lambda storage, loc:storage)
tsp50.load_state_dict(state)

if USE_CUDA:
        tsp50 = tsp50.cuda()

tsp50_dataset = pickle.load(open("test_data_set_50", "rb"))

tsp50_loader = DataLoader(tsp50_dataset, batch_size=128, shuffle=False, num_workers=1)

tour_lengths = np.zeros(1)

for batch_id, sample_batch in enumerate(tsp50_loader):

        inputs = Variable(sample_batch)

        if USE_CUDA:
                inputs = inputs.cuda()

        R, probs, actions, action_idxs, critic_evals = tsp50(inputs, T=2.2)

        if (tour_lengths == np.zeros(1)).all():
                tour_lengths = R.data.numpy()
        else:
                tour_lengths = np.append(tour_lengths, R.data.numpy())

print (tour_lengths.mean())
print (tour_lengths)

file = open('results_greedy_tsp50', 'wb')
pickle.dump(tour_lengths, file)
file.close()