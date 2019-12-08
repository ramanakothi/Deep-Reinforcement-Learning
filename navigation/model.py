import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dense1=nn.Linear(state_size,64)
        self.dense2=nn.Linear(64,252)
        self.dense3=nn.Linear(252,124)
        self.dense4=nn.Linear(124,action_size)
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x=F.relu(self.dense1(state))
        x=F.relu(self.dense2(x))
        x=F.relu(self.dense3(x))
        x=self.dense4(x)
        return x
    
class QNetworkDuel(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkDuel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dense1_1=nn.Linear(state_size,64)
        self.dense1_2=nn.Linear(state_size,64)
        self.dense2_1=nn.Linear(64,64)
        self.dense2_2=nn.Linear(64,64)
        self.dense3_1=nn.Linear(64,64)
        self.dense3_2=nn.Linear(64,64)
        self.dense4_1=nn.Linear(64,1)
        self.dense4_2=nn.Linear(64,action_size)
        self.action_size=action_size
        
    def forward(self, state):
        
        x1=F.relu(self.dense1_1(state))
        x1=F.relu(self.dense2_1(x1))
        x1=F.relu(self.dense3_1(x1))
        v=self.dense4_1(x1).expand(x1.size(0),self.action_size)
        x2=F.relu(self.dense1_2(state))
        x2=F.relu(self.dense2_2(x2))
        x2=F.relu(self.dense3_2(x2))
        adv=self.dense4_2(x2)
        x3=v+adv-adv.mean(1).unsqueeze(1).expand(x1.size(0),self.action_size)
        
        return x3