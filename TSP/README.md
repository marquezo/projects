# Code for course project for COMP767 Reinforcement Learning 
Solving the Travelling Salesman problem using RL and Pointer Networks. We reproduced the main results of the paper "Neural Combinatorial Optimization with Reinforcement Learning" by Bello et al. and extended their conclusions by empirically demonstrating that TSP agents generalize well around their neighbourhood: they can find good solutions to graphs larger or smaller than those in which the agent was trained.

We borrowed initial setup from https://github.com/higgsfield/np-hard-deep-reinforcement-learning

A TSP agent can be trained via: python main.py [learning rate for agent] [number of points in graph] [number of epochs]

We generate 1,000,000 graphs to constitute 1 epoch of training data.

Example: python main.py 0.001 20 5 (to train a TSP20 agent starting with learning rate 0.001 and for 5 epochs)

At the end of each mini-batch, the model parameters will be saved as tsp_model_general.pt and the tour lengths after each minibatch will be saved as tour_length_general




