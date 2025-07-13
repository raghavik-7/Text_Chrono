import numpy as np
from .rnn_cell import VanillaRNNCell
from .lstm_cell import LSTMCell

class StackedRNN:
    def __init__(self, input_size, hidden_size, num_layers):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(VanillaRNNCell(in_size, hidden_size))
    def forward(self, x_seq, h0=None, x_idx_seq=None):
        # x_seq: (seq_len, input_size, batch_size)
        seq_len, _, batch_size = x_seq.shape
        hs_all = []
        x = x_seq
        h_states = []
        for i, layer in enumerate(self.layers):
            h0_i = None if h0 is None else h0[i]
            x_idx = x_idx_seq if (i == 0) else None
            hs = layer.forward(x, h0=h0_i, x_idx_seq=x_idx)
            x = np.stack(hs, axis=0).astype(np.float32)  # (seq_len, hidden_size, batch_size)
            h_states.append(hs)
        self.last_x_seq = x_seq
        self.last_x_idx_seq = x_idx_seq
        return hs  # output of last layer
    def backward(self, dhs, clip=5.0):
        # dhs: list of gradients for last layer
        for i in reversed(range(self.num_layers)):
            layer = self.layers[i]
            layer.backward(dhs, clip=clip)
            dhs = [np.dot(layer.U.T, d) for d in dhs]  # propagate to previous layer
        return dhs
    def step(self, lr=0.01):
        for layer in self.layers:
            layer.step(lr)

class StackedLSTM:
    def __init__(self, input_size, hidden_size, num_layers):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(LSTMCell(in_size, hidden_size))
    def forward(self, x_seq, h0=None, c0=None):
        # x_seq: (seq_len, input_size, batch_size)
        seq_len, _, batch_size = x_seq.shape
        x = x_seq
        hs_all = []
        cs_all = []
        for i, layer in enumerate(self.layers):
            h0_i = None if h0 is None else h0[i]
            c0_i = None if c0 is None else c0[i]
            hs, cs = layer.forward(x, h0=h0_i, c0=c0_i)
            x = np.stack(hs, axis=0).astype(np.float32)  # (seq_len, hidden_size, batch_size)
            hs_all.append(hs)
            cs_all.append(cs)
        return hs, cs
    def backward(self, dhs, dcs=None, clip=5.0):
        # dhs, dcs: list of gradients for last layer
        for i in reversed(range(self.num_layers)):
            layer = self.layers[i]
            layer.backward(dhs, dcs, clip=clip)
            dhs = [np.dot(layer.W_o.T[:, :self.hidden_size], d) for d in dhs]  # propagate to previous layer
        return dhs, dcs
    def step(self, lr=0.01):
        for layer in self.layers:
            layer.step(lr) 