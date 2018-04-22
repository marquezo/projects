from CombinatorialRL import CombinatorialRL
from torch.utils.data import DataLoader
from inference import *
import torch
import pickle
import sys
import numpy as np

train_size = 1000000
val_size = 10000

embedding_size = 128
hidden_size    = 128
n_glimpses = 1
tanh_exploration = 10
use_tanh = True
seq_len=int(sys.argv[1])

max_grad_norm = 1.
USE_CUDA = torch.cuda.is_available()

pretrained_model_loc = sys.argv[2]
dataset_loc = sys.argv[3]
results_saved_loc = sys.argv[4]
temperature =  float(sys.argv[5])
number_solutions =  int(sys.argv[6])

#EXAMPLE: python inference_rl_pretraining_sampling.py 20 "Model/TSP20/tsp_model_TSP20_001.pt" test_data_set_20 results_sampling_multiple_one 2.0 128

model = CombinatorialRL(
        embedding_size,
        hidden_size,
        seq_len,
        n_glimpses,
        tanh_exploration,
        use_tanh,
        attention="Bahdanau",
        use_cuda=USE_CUDA)

if(USE_CUDA):
	state = torch.load(pretrained_model_loc)
else:
	state = torch.load(pretrained_model_loc, map_location=lambda storage, loc:storage)

model.load_state_dict(state)

dataset = pickle.load(open(dataset_loc, "rb"))

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

sum_solutions = 0.0

tours = []
for batch_id, sample_batch in enumerate(dataloader):

        sample_batch = sample_batch.squeeze()

        if USE_CUDA:
                sample_batch = sample_batch.cuda()
        lengths = []

        for i in range(number_solutions/128):
                soln_sampling, tour_len  = sample_solution(sample_batch, model, 128, T=temperature)

        lengths.append(tour_len.data[0])
        idx = np.argmin(lengths)
        min_tour_len = lengths[idx]

        tours.append(min_tour_len)
        print ("batch id {}, result: {}".format(batch_id, min_tour_len))

tf = open(results_saved_loc, 'wb')
pickle.dump(tours, tf)
tf.close()

print "Average length of tour: ", (np.mean(tours))