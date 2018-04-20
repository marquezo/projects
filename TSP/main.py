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

    train_20_dataset = TSPDataset(20, train_size)
    val_20_dataset = TSPDataset(20, val_size)

    tsp_20_model = CombinatorialRL(
        embedding_size,
        hidden_size,
        20,
        n_glimpses,
        tanh_exploration,
        use_tanh,
        attention="Bahdanau",
        use_cuda=USE_CUDA)

    lr_actor = float(sys.argv[1])
    lr_critic = float(sys.argv[2])

    tsp_20_train = TrainModel(tsp_20_model,
                              train_20_dataset,
                              val_20_dataset,
                              threshold=3.99, use_cuda=USE_CUDA)

    tsp_20_train.train_and_validate(n_epochs=5,
                                    lr_actor=lr_actor,
                                    lr_critic=lr_critic)


