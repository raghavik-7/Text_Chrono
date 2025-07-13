import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale = np.sqrt(input_size + hidden_size)
        # Forget gate
        self.W_f = (np.random.randn(hidden_size, input_size + hidden_size).astype(np.float32)) / scale
        self.b_f = np.zeros((hidden_size, 1), dtype=np.float32)
        # Input gate
        self.W_i = (np.random.randn(hidden_size, input_size + hidden_size).astype(np.float32)) / scale
        self.b_i = np.zeros((hidden_size, 1), dtype=np.float32)
        # Candidate memory
        self.W_c = (np.random.randn(hidden_size, input_size + hidden_size).astype(np.float32)) / scale
        self.b_c = np.zeros((hidden_size, 1), dtype=np.float32)
        # Output gate
        self.W_o = (np.random.randn(hidden_size, input_size + hidden_size).astype(np.float32)) / scale
        self.b_o = np.zeros((hidden_size, 1), dtype=np.float32)
        # For storing gradients
        self.dW_f = np.zeros_like(self.W_f)
        self.db_f = np.zeros_like(self.b_f)
        self.dW_i = np.zeros_like(self.W_i)
        self.db_i = np.zeros_like(self.b_i)
        self.dW_c = np.zeros_like(self.W_c)
        self.db_c = np.zeros_like(self.b_c)
        self.dW_o = np.zeros_like(self.W_o)
        self.db_o = np.zeros_like(self.b_o)

    def forward(self, x_seq, h0=None, c0=None):
        # x_seq: (seq_len, input_size, batch_size)
        seq_len = x_seq.shape[0]
        batch_size = x_seq.shape[2]
        if h0 is None:
            h0 = np.zeros((self.hidden_size, batch_size), dtype=np.float32)
        if c0 is None:
            c0 = np.zeros((self.hidden_size, batch_size), dtype=np.float32)
        hs = [h0]
        cs = [c0]
        self.cache = []
        for t in range(seq_len):
            x = x_seq[t]
            concat = np.vstack((hs[-1], x)).astype(np.float32)
            f_t = self._sigmoid(np.dot(self.W_f, concat) + self.b_f).astype(np.float32)
            i_t = self._sigmoid(np.dot(self.W_i, concat) + self.b_i).astype(np.float32)
            c_hat_t = np.tanh(np.dot(self.W_c, concat) + self.b_c).astype(np.float32)
            c_t = f_t * cs[-1] + i_t * c_hat_t
            o_t = self._sigmoid(np.dot(self.W_o, concat) + self.b_o).astype(np.float32)
            h_t = o_t * np.tanh(c_t)
            hs.append(h_t.astype(np.float32))
            cs.append(c_t.astype(np.float32))
            self.cache.append((x, hs[-2], cs[-2], f_t, i_t, c_hat_t, c_t, o_t, h_t, concat))
        return hs[1:], cs[1:]

    def backward(self, dhs, dcs=None, clip=5.0):
        seq_len = len(dhs)
        batch_size = dhs[0].shape[1]
        if dcs is None:
            dcs = [np.zeros((self.hidden_size, batch_size), dtype=np.float32) for _ in range(seq_len)]
        dW_f = np.zeros_like(self.W_f)
        db_f = np.zeros_like(self.b_f)
        dW_i = np.zeros_like(self.W_i)
        db_i = np.zeros_like(self.b_i)
        dW_c = np.zeros_like(self.W_c)
        db_c = np.zeros_like(self.b_c)
        dW_o = np.zeros_like(self.W_o)
        db_o = np.zeros_like(self.b_o)
        dh_next = np.zeros((self.hidden_size, batch_size), dtype=np.float32)
        dc_next = np.zeros((self.hidden_size, batch_size), dtype=np.float32)
        dxs = []  # To store gradients w.r.t. input (embedding) at each time step
        for t in reversed(range(seq_len)):
            (x, h_prev, c_prev, f_t, i_t, c_hat_t, c_t, o_t, h_t, concat) = self.cache[t]
            dh = dhs[t] + dh_next
            dc = dcs[t] + dc_next
            do = dh * np.tanh(c_t)
            do_pre = do * o_t * (1 - o_t)
            dc_t = dh * o_t * (1 - np.tanh(c_t) ** 2) + dc
            df = dc_t * c_prev
            df_pre = df * f_t * (1 - f_t)
            di = dc_t * c_hat_t
            di_pre = di * i_t * (1 - i_t)
            dc_hat = dc_t * i_t
            dc_hat_pre = dc_hat * (1 - c_hat_t ** 2)
            dW_f += np.dot(df_pre, concat.T)
            db_f += np.sum(df_pre, axis=1, keepdims=True)
            dW_i += np.dot(di_pre, concat.T)
            db_i += np.sum(di_pre, axis=1, keepdims=True)
            dW_c += np.dot(dc_hat_pre, concat.T)
            db_c += np.sum(dc_hat_pre, axis=1, keepdims=True)
            dW_o += np.dot(do_pre, concat.T)
            db_o += np.sum(do_pre, axis=1, keepdims=True)
            dconcat = (
                np.dot(self.W_f.T, df_pre) +
                np.dot(self.W_i.T, di_pre) +
                np.dot(self.W_c.T, dc_hat_pre) +
                np.dot(self.W_o.T, do_pre)
            )
            dh_next = dconcat[:self.hidden_size, :]
            dc_next = dc_t * f_t
            # Compute gradient w.r.t. input (embedding vector)
            dx = dconcat[self.hidden_size:, :]  # (input_size, batch_size)
            dxs.insert(0, dx)  # Insert at beginning to maintain time order
        # Gradient clipping
        for grad in [dW_f, db_f, dW_i, db_i, dW_c, db_c, dW_o, db_o]:
            np.clip(grad, -clip, clip, out=grad)
        self.dW_f, self.db_f = dW_f, db_f
        self.dW_i, self.db_i = dW_i, db_i
        self.dW_c, self.db_c = dW_c, db_c
        self.dW_o, self.db_o = dW_o, db_o
        return dxs  # List of (input_size, batch_size) gradients, one per time step

    def step(self, lr=0.01):
        print(f"[DEBUG] id(self.W_f) before: {id(self.W_f)}")
        self.W_f[...] -= lr * self.dW_f
        self.b_f[...] -= lr * self.db_f
        self.W_i[...] -= lr * self.dW_i
        self.b_i[...] -= lr * self.db_i
        self.W_c[...] -= lr * self.dW_c
        self.b_c[...] -= lr * self.db_c
        self.W_o[...] -= lr * self.dW_o
        self.b_o[...] -= lr * self.db_o
        print(f"[DEBUG] id(self.W_f) after: {id(self.W_f)}")

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) 