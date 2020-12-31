import torch.nn as nn
import torch.nn.functional as F
from utils.noisy_linear import NoisyLinear


class NoisyRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NoisyRNNAgent, self).__init__()
        self.args = args

        self.fc1 = NoisyLinear(input_shape, args.rnn_hidden_dim, True, args.device)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = NoisyLinear(args.rnn_hidden_dim, args.n_actions, True, args.device)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.u_w.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
