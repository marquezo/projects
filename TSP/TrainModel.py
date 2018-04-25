
from IPython.display import clear_output
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pickle


class TrainModel:
    def __init__(self, model, train_dataset, val_dataset, batch_size=128, threshold=None, max_grad_norm=2.,
                 use_cuda=False):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.max_grad_norm = max_grad_norm
        self.use_cuda = use_cuda

        self.train_tour = []
        self.val_tour = []

        self.epochs = 0
        self.beta = 0.9  # For exponential moving average if no critic is used

    def train_and_validate(self, n_epochs, lr_actor, lr_critic, use_critic=False):

        self.actor_optim = optim.Adam(self.model.actor.parameters(), lr=lr_actor)

        if use_critic:
            self.critic_optim = optim.Adam(self.model.critic.parameters(), lr=lr_critic)
            critic_loss_criterion = torch.nn.MSELoss()
        else:
            critic_exp_mvg_avg = torch.zeros(1)

        if self.use_cuda and use_critic:
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

        for epoch in range(n_epochs):
            for batch_id, sample_batch in enumerate(self.train_loader):
                self.model.train()

                inputs = Variable(sample_batch)

                if self.use_cuda:
                    inputs = inputs.cuda()

                # Model is combinatorial
                R, probs, actions, actions_idxs, values = self.model(inputs)

                if batch_id == 0:
                    critic_exp_mvg_avg = R.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * self.beta) + ((1. - self.beta) * R.mean())

                # Vector of length equal to the mini-batch size: Q(s,a) - V(s)
                if use_critic:
                    advantage = R.unsqueeze(1) - values
                else:
                    advantage = R - critic_exp_mvg_avg

                    # print ("Advantage function: ", R.mean() - values.mean())

                logprobs = 0
                for prob in probs:
                    logprob = torch.log(prob)
                    logprobs += logprob

                # For Pytorch 3.0
                if logprobs.data[0] < -1000:
                    print (logprobs.data[0])
                    logprobs = Variable(torch.FloatTensor([0.]), requires_grad=True)

                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(self.model.actor.parameters(),
                                              float(self.max_grad_norm), norm_type=2)

                self.actor_optim.step()

                if use_critic:
                    # Do critic gradient descent
                    self.critic_optim.zero_grad()
                    loss_critic = critic_loss_criterion(values, R.unsqueeze(1))
                    loss_critic.backward()
                    torch.nn.utils.clip_grad_norm(self.model.critic.parameters(),
                                                  float(self.max_grad_norm), norm_type=2)
                    self.critic_optim.step()
                else:
                    critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                self.train_tour.append(R.mean().data[0])

                if batch_id % 100 == 0:
                    #self.plot(self.epochs)
                    torch.save(self.model.state_dict(), 'tsp_model_general.pt')

                    file = open('tour_length_general', 'wb')
                    pickle.dump(self.train_tour, file)
                    file.close()

                    if use_critic:
                        print ("Epoch {}, Batch {}: Actor says {} | Critic says {}".format(epoch, batch_id, R.mean().data[0],
                                                                                    values.mean().data[0]))
                    else:
                        print ("Epoch {}, Batch {}: Actor says {}".format(epoch, batch_id, R.mean().data[0]))

            #                 if batch_id % 100 == 0:

            #                     self.model.eval()
            #                     for val_batch in self.val_loader:
            #                         inputs = Variable(val_batch)

            #                         if USE_CUDA:
            #                             inputs = inputs.cuda()

            #                         R, probs, actions, actions_idxs, values = self.model(inputs)
            #                         self.val_tour.append(R.mean().data[0])

            if self.threshold and self.train_tour[-1] < self.threshold:
                print ("EARLY STOPPAGE!")
                break

            self.epochs += 1

    def plot(self, epoch):
        clear_output(True)
        plt.figure(figsize=(30, 5))
        plt.subplot(131)
        plt.title('train tour length: epoch %s reward %s' % (
        epoch, self.train_tour[-1] if len(self.train_tour) else 'collecting'))
        plt.plot(self.train_tour)
        plt.grid()
        #         plt.subplot(132)
        #         plt.title('val tour length: epoch %s reward %s' % (epoch, self.val_tour[-1] if len(self.val_tour) else 'collecting'))
        #         plt.plot(self.val_tour)
        #         plt.grid()
        plt.show()