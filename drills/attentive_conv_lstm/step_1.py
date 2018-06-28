import torch
from torch import nn
import torch.nn.functional as F

'''
    Step 1
    - Apply attentive equations to the AttentiveConvLSTMCell module
      your "new" xt should be obtained by multiplying xt with at
      where at is the result of the attentive softmax
      
    - You will need three new convolutional layers
'''

class AttentiveConvLSTMCell(nn.Module):
    def __init__(self, in_channels, h_channels, k):
        super(AttentiveConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        assert(k % 2 == 1)
        self.k = k

        self.i2h = self.conv(in_channels, h_channels*5)
        self.h2h = self.conv(h_channels, h_channels*5)

        self.init_weights()

    def conv(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, self.k, padding=(self.k-1)//2)

    def init_weights(self):
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2h.weight)
        nn.init.constant_(self.i2h.bias, .0)
        nn.init.constant_(self.h2h.bias, .0)

    def forward(self, xt, state):
        # Conv-LSTM
        all_input_sums = self.i2h(xt) + self.h2h(state[0])
        sigmoid_chunk = F.sigmoid(all_input_sums[:, :3*self.h_channels])
        it, ft, ot = sigmoid_chunk.split(self.h_channels, 1)
        gt = F.tanh(all_input_sums[:, 4*self.h_channels:])
        ct = ft * state[1] + it * gt
        ht = ot * F.tanh(ct)

        output = ht.unsqueeze(1)
        state = (ht, ct)
        return output, state
