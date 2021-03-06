from CombinatorialRL import CombinatorialRL
from torch.utils.data import DataLoader
from inference import *
import torch
import pickle

train_size = 1000000
val_size = 10000

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

# state = torch.load('Model/TSP20/tsp_model_TSP20_001.pt', map_location=lambda storage, loc:storage)
# tsp20.load_state_dict(state)

if USE_CUDA:
        tsp50 = tsp50.cuda()

tsp50_dataset = pickle.load(open("test_data_set_50", "rb"))

tsp50_loader = DataLoader(tsp50_dataset, batch_size=1, shuffle=False, num_workers=1)

tour_lengths = []

for batch_id, sample_batch in enumerate(tsp50_loader):
        sample_batch = sample_batch.squeeze()

        if USE_CUDA:
                sample_batch = sample_batch.cuda()
        #1e-5 was used by authors
        soln_sampling, min_tour_len = active_search(sample_batch, tsp50, 1280000, 128, lr=1e-3)
        tour_lengths.append(min_tour_len)
        print ("batch id {}, result: {}".format(batch_id, min_tour_len))

file = open('results_only_active_search_tsp50', 'wb')
pickle.dump(tour_lengths, file)
file.close()
