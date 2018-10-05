from CombinatorialRL import CombinatorialRL
from torch.utils.data import DataLoader
from inference import *
import numpy as np
import torch
import pickle


embedding_size = 128
hidden_size    = 128
n_glimpses = 1
tanh_exploration = 10
use_tanh = True
seq_len=35

max_grad_norm = 1.
USE_CUDA = False

tsp = CombinatorialRL(
        embedding_size,
        hidden_size,
        seq_len,
        n_glimpses,
        tanh_exploration,
        use_tanh,
        attention="Bahdanau",
        use_cuda=USE_CUDA)

state = torch.load('Model/TSP35/tsp_model_TSP35_0005.pt', map_location=lambda storage, loc:storage)
tsp.load_state_dict(state)

dataset = pickle.load(open("test_data_set_50", "rb"))

data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

tours = []

for batch_id, sample_batch in enumerate(data_loader):

        if batch_id <10:

                sample_batch = sample_batch.squeeze()
                lengths = []

                for i in range(100):
                        soln_sampling, tour_len = sample_solution(sample_batch, tsp, 128, T=1.0)
                        lengths.append(tour_len.data[0])

                min_tour_len = np.min(lengths)

                print "batch id {}, result: {}".format(batch_id, min_tour_len)
                tours.append(min_tour_len)

print ("Result: ", np.mean(tours))
