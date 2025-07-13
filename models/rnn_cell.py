import numpy as np

class VanillaRNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = (np.random.randn(hidden_size, input_size).astype(np.float32)) / np.sqrt(input_size)
        self.U = (np.random.randn(hidden_size, hidden_size).astype(np.float32)) / np.sqrt(hidden_size)
        self.b = np.zeros((hidden_size, 1), dtype=np.float32)
        # For storing gradients
        self.dW = np.zeros_like(self.W)
        self.dU = np.zeros_like(self.U)
        self.db = np.zeros_like(self.b)

    def forward(self, x_seq, h0=None):
        # x_seq: (seq_len, input_size, batch_size)
        seq_len = x_seq.shape[0]
        batch_size = x_seq.shape[2]
        if h0 is None:
            h0 = np.zeros((self.hidden_size, batch_size), dtype=np.float32)
        hs = [h0]
        self.cache = []
        for t in range(seq_len):
            x = x_seq[t]
            h_prev = hs[-1]
            h = np.tanh(np.dot(self.W, x) + np.dot(self.U, h_prev) + self.b).astype(np.float32)
            hs.append(h)
            # Store x for backward
            self.cache.append((x, h_prev, h))
        return hs[1:]

    def backward(self, dhs, clip=5.0):
        seq_len = len(dhs)
        batch_size = dhs[0].shape[1]
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)
        dh_next = np.zeros((self.hidden_size, batch_size), dtype=np.float32)
        dxs = []  # To store gradients w.r.t. input (embedding) at each time step
        for t in reversed(range(seq_len)):
            x, h_prev, h = self.cache[t]
            dh = dhs[t] + dh_next
            dtanh = (1 - h ** 2) * dh
            dW += np.dot(dtanh, x.T)  # Update input-to-hidden weights
            dU += np.dot(dtanh, h_prev.T)
            db += np.sum(dtanh, axis=1, keepdims=True)
            dx = np.dot(self.W.T, dtanh)  # (input_size, batch_size)
            dxs.insert(0, dx)  # Insert at beginning to maintain time order
            dh_next = np.dot(self.U.T, dtanh)
        # Gradient clipping
        for grad in [dW, dU, db]:
            np.clip(grad, -clip, clip, out=grad)
        self.dW, self.dU, self.db = dW, dU, db
        return dxs  # List of (input_size, batch_size) gradients, one per time step

    def step(self, lr=0.01):
        print(f"[DEBUG] id(self.W) before: {id(self.W)}")
        self.W[...] -= lr * self.dW
        self.U[...] -= lr * self.dU
        self.b[...] -= lr * self.db
        print(f"[DEBUG] id(self.W) after: {id(self.W)}") 