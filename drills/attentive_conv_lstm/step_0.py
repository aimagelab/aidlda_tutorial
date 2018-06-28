import torch
from torch import nn
import torch.nn.functional as F

'''
    Step 0
    - Build a nn.Module that defines the AttentiveConvLSTMCell
    
    - Constructor should take as input:
        - The number of input channels, in_channels
        - The number of channels of hidden states, h_channels
        - The size of all kernels, k
      furthermore, it should correctly initialize the parameters via a init_weights function.
      
    - forward() should receive:
        - The current input, xt (b_s, in_channels, h, w)
        - The current state, state: a tuple containing h_t and c_t (b_s, h_channels, h, w)
      and return:
        - The next output, (b_s, 1, h_channels, h, w)
        - The next state: a tuple containing h_{t+1} and c_{t+1} (b_s, h_channels, h, w)
      at this stage, do not take into account the attentive part. Use ordinary LSTM equations and convolutions.
'''

class AttentiveConvLSTMCell(nn.Module):
    def __init__(self, in_channels, h_channels, k):
        super(AttentiveConvLSTMCell, self).__init__()

    def init_weights(self):
        pass

    def forward(self, xt, state):
        pass