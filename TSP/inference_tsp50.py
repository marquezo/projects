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
USE_CUDA = False

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

tsp50_dataset = pickle.load(open("test_data_set_50", "rb"))

tsp50_loader = DataLoader(tsp50_dataset, batch_size=1, shuffle=False, num_workers=1)

sum_solutions = 0.0

for batch_id, sample_batch in enumerate(tsp50_loader):
        sample_batch = sample_batch.squeeze()
        soln_sampling = sample_solution(sample_batch, tsp50, 128)
        print reward_single_input(soln_sampling)
        sum_solutions +=  soln_sampling

print "Result: ", (sum_solutions/1000.0)