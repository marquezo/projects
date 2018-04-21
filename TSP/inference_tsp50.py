from CombinatorialRL import CombinatorialRL
import torch

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

state = torch.load('Model/TSP50/tsp_model_T50_001.pt')
tsp50.load_state_dict(state)