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
seq_len=20

max_grad_norm = 1.
USE_CUDA = False

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

tsp20_dataset = pickle.load(open("test_data_set_20", "rb"))

tsp20_loader = DataLoader(tsp20_dataset, batch_size=1, shuffle=False, num_workers=1)

sum_tour_length = 0.0
tours = []
for batch_id, sample_batch in enumerate(tsp20_loader):
        sample_batch = sample_batch.squeeze()
	lengths = []
	solns = []
	for i in range(10):
        	soln_sampling, tour_len = sample_solution(sample_batch, tsp20, 128, T=2.0)
		lengths.append(tour_len.data[0])
	idx = np.argmin(lengths)
	min_tour_len = lengths[idx]
        sum_tour_length +=  min_tour_len
	tours.append(min_tour_len)
        print "batch id {}, result: {}".format(batch_id, min_tour_len)
tf = open('tours_tsp20', 'wb')
pickle.dump(tours,tf)
tf.close()

print "Result: ", (sum_tour_length/1000.0)
