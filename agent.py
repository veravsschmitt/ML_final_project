# Implementation of a Deep-Q-Learning agent
# base copied from topics in cs project, based on https://github.com/krazyness/CRBot-public/blob/main/env.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# this class inherits from pytorchs nn.module, a basic class for neural networks and defines a nn
class DQN(nn.Module):

    # define layers and components of the model 
    # input_dim: dimension of the state vector we give the model
    # output_dim: number of actions for the agent to do
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        # layers of the network 
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.ReLU(),                 
            nn.Linear(64, output_dim)  
        )

    # define computation that happens on input tensors
    def forward(self, x):
        return self.net(x)

# this class defines the actual agent   
class DQNAgent:
    def __init__(self, state_size, action_size):

        self.model = DQN(state_size, action_size)                       #constructing model that will learn
        self.target_model = DQN(state_size, action_size)                #constructing second model for stabilization (this always gets updated with the actual model)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  #use the adam optimizer for training
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)                               #replay storage where we can store 10000 experiences
        self.gamma = 0.95
        self.epsilon = 1.0                                              #probability of doing a random action (exploration)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.action_size = action_size                                  #amount of possible actions

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done):                              #add the expirence to the replay storage
        self.memory.append((s, a, r, s2, done))

    # implementation of epsilon greedy algorithm
    # with the probability of epsiolon we decide if we want to do a random action or the best action according to our current network (exploration or exploitation)
    def act(self, state):
        if random.random() < self.epsilon:
            # do a random action 
            return random.randrange(self.action_size)
        
        # do the best action according to the current network
        state = torch.FloatTensor(state).unsqueeze(0)               #unsqueez to make dimensions match, converts input tensor to float
        with torch.no_grad():                                       #only forward pass, no updating or improving the model, only get q values for the states -> saves memory and speeds this up
            q_values = self.model(state)                            #output of the model is a q-value for each possible action to do in the input stage ()
        return q_values.argmax().item()                             #action with the highest q value is the best possible action


    #training of the agent with collected experience from the replay memory
    def replay(self, batch_size):

        # no training possible if not enough data in replay memory yet to train
        if len(self.memory) < batch_size:
            return
        
        # randomly picking experiences from the replay memory to train
        batch = random.sample(self.memory, batch_size)

        # training loop to do for each experience in replay memory
        for state, action, reward, next_state, done in batch:

            #calculation of the target q value (this is the q value we are trying to reach)
            target = reward                                                 # target value is the current reward to start with
            if not done:
                target += self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state)))
                # if the state is not terminal: add the maximal ecpected q value of the next state to the target
                # use target_model for more stabilized values
                # use gamma to reduce the influence of the q vaules of the next state
            
            #predict q-value for current state
            target_f = self.model(torch.FloatTensor(state))                 # get q values from current model for current state
            target_f = target_f.clone()                                     # maka an independent copy                                   
            target_f[action] = float(target.item())                                # set the q-value of the current action to target

            prediction = self.model(torch.FloatTensor(state))[action]       # get predicted q value from the current model for the current action
            loss = self.criterion(prediction, target_f[action].detach())    # loss function calculates how far prediction q value is from target q value

            # updates network weigths with help of gradients 
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()

        # adjust exploration rate -> with more training the agent should do less random acts and more acts based on future experiences
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    # loads the trained model (loading of learned weights, no gradient calculations anymore, ...) from the file filename
    def load(self, filename):

        # Look in models/ directory by default (if path is not absolute)
        path = filename
        if not os.path.isabs(filename):
            path = os.path.join("models", filename)

        # load the saved weigths from training into the current model
        self.model.load_state_dict(torch.load(path))

        # set the model to evaluating mode 
        self.model.eval()
        
        print(f"Loaded model weights from {path}")