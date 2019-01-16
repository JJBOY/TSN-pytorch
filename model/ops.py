import torch
import math


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        self.shape = input.size()
        if self.consensus_type == 'avg':
            output = input.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input
        else:
            output = None
        return output
