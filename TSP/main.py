from TSPDataset import TSPDataset
from CombinatorialRL import CombinatorialRL
from TrainModel import TrainModel
import sys
import torch

train_size = 1000000
val_size = 10000

embedding_size = 128
hidden_size    = 128
n_glimpses = 1
tanh_exploration = 10
use_tanh = True

max_grad_norm = 1.
USE_CUDA = torch.cuda.is_available()

if __name__ == '__main__':

    lr_actor = float(sys.argv[1])
    #lr_critic = float(sys.argv[2])
    seq_len = int(sys.argv[2])
    num_epochs = int(sys.argv[3])

    train_dataset = TSPDataset(seq_len, train_size)
    val_dataset = TSPDataset(seq_len, val_size)

    tsp_model = CombinatorialRL(
        embedding_size,
        hidden_size,
        seq_len,
        n_glimpses,
        tanh_exploration,
        use_tanh,
        attention="Bahdanau",
        use_cuda=USE_CUDA)

    tsp_train = TrainModel(tsp_model,
                              train_dataset,
                              val_dataset,
                              threshold=3.99, use_cuda=USE_CUDA)

    tsp_train.train_and_validate(n_epochs=num_epochs,
                                    lr_actor=lr_actor,
                                    lr_critic=None)


