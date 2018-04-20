
from IPython.display import clear_output
#from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class TrainModel:
    def __init__(self, model, train_dataset, val_dataset, batch_size=128, threshold=None, max_grad_norm=2., use_cuda=False):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold
        self.use_cuda = use_cuda

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.max_grad_norm = max_grad_norm

        self.train_tour = []
        self.val_tour   = []

        self.epochs = 0

    def train_and_validate(self, n_epochs, lr_actor, lr_critic, scheduler_step, scheduler_gamma):

        self.actor_optim   = optim.Adam(self.model.actor.parameters(), lr=lr_actor)
        self.critic_optim  = optim.Adam(self.model.critic.parameters(), lr=lr_critic)
        self.scheduler_actor = optim.lr_scheduler.StepLR(self.actor_optim, step_size=scheduler_step, gamma=scheduler_gamma)
        self.scheduler_critic = optim.lr_scheduler.StepLR(self.critic_optim, step_size=scheduler_step, gamma=scheduler_gamma)

        critic_exp_mvg_avg = torch.zeros(1)
        critic_loss_criterion = torch.nn.MSELoss()

        if self.use_cuda:
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

        for epoch in range(n_epochs):
            for batch_id, sample_batch in enumerate(self.train_loader):
                self.model.train()

                self.scheduler_actor.step()
                self.scheduler_critic.step()

                inputs = Variable(sample_batch)

                if self.use_cuda:
                    inputs = inputs.cuda()

                # Model is combinatorial
                R, probs, actions, actions_idxs, values = self.model(inputs)

                # if batch_id == 0:
                #     critic_exp_mvg_avg = R.mean()
                # else:
                #     critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

                # Vector of length equal to the mini-batch size: Q(s,a) - V(s)
                # advantage = R - critic_exp_mvg_avg
                advantage = R.unsqueeze(1) - values

                # print ("Advantage function: ", R.mean() - values.mean())

                logprobs = 0
                for prob in probs:
                    logprob = torch.log(prob)
                    logprobs += logprob

                # logprobs[logprobs < -1000] = 0. #Works with PyTorch 2.0

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

                # Do critic gradient descent
                self.critic_optim.zero_grad()
                loss_critic = critic_loss_criterion(values, R.unsqueeze(1))
                loss_critic.backward()
                torch.nn.utils.clip_grad_norm(self.model.critic.parameters(),
                                              float(self.max_grad_norm), norm_type=2)

                self.critic_optim.step()
                # print ("Critic's loss: ", loss_critic.data[0])

                # critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                self.train_tour.append(R.mean().data[0])

                if batch_id % 100 == 0:
                    # self.plot(self.epochs)
                    print ("Epoch {}, Batch {}: Actor says {} | Critic says {}".format(epoch, batch_id, R.mean().data[0], values.mean().data[0]) )

            #                 if batch_id % 100 == 0:

            #                     self.model.eval()
            #                     for val_batch in self.val_loader:
            #                         inputs = Variable(val_batch)

            #                         if USE_CUDA:
            #                             inputs = inputs.cuda()

            #                         R, probs, actions, actions_idxs, values = self.model(inputs)
            #                         self.val_tour.append(R.mean().data[0])

            if self.threshold and self.train_tour[-1] < self.threshold:
                print "EARLY STOPPAGE!"
                break

            self.epochs += 1

    def plot(self, epoch):
        clear_output(True)
        plt.figure(figsize=(20 ,5))
        plt.subplot(131)
        plt.title('train tour length: epoch %s reward %s' %
        (epoch, self.train_tour[-1] if len(self.train_tour) else 'collecting'))
        plt.plot(self.train_tour)
        plt.grid()
        #         plt.subplot(132)
        #         plt.title('val tour length: epoch %s reward %s' % (epoch, self.val_tour[-1] if len(self.val_tour) else 'collecting'))
        #         plt.plot(self.val_tour)
        #         plt.grid()
        plt.show()