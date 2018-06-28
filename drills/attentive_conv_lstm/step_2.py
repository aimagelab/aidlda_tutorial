import torch
from torch import nn
import torch.nn.functional as F

class AttentiveConvLSTMCell(nn.Module):
    def __init__(self, in_channels, h_channels, k):
        super(AttentiveConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        assert(k % 2 == 1)
        self.k = k

        self.i2ha = self.conv(in_channels, h_channels)
        self.h2ha = self.conv(h_channels, h_channels)
        self.va = self.conv(h_channels, 1)

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
        # Attentive part
        pzt = F.tanh(self.i2ha(xt) + self.h2ha(state[0]))
        zt = self.va(pzt)
        zt_shape = zt.shape
        at = F.softmax(zt.view(zt.shape[0], zt.shape[1], -1), -1).view(zt_shape)
        xt = xt * at

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

'''
    Step 2
    - Now that you have a working implementation of an AttentiveConvLSTMCell,
      build the complete Attentive ConvLSTM layer.
      
    - This will be a new nn.Module.
      Given an input image, it should be fed to the LSTM cell at all iterations.
      The output sequence should be returned.

    - Constructor should take as input:
        - The number of input channels, in_channels
        - The number of channels of hidden states, h_channels
        - The size of all kernels, k
        - The number of iterations the LSTM cell should make, seq_len
        
    - forward() should receive:
        - The input image, input (b_s, in_channels, h, w)
      and return:
        - The sequence of outputs, (b_s, seq_len, h_channels, h, w)    
      when called, it should also initialize the hidden states of the cell via an init_state function           
'''
class AttentiveLSTM(nn.Module):
    def __init__(self, in_channels, h_channels, k, seq_len):
        super(AttentiveLSTM, self).__init__()

    def init_state(self, b_s, h, w, device):
        pass

    def init_weights(self):
        pass

    def forward(self, input):
        pass