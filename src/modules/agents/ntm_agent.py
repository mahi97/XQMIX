import torch
import torch.nn.functional as F
from torch.nn.parallel import data_parallel
from torch import nn
from torch import optim
import numpy as np
import copy


def _convolve(wg, sg, batch_size):
    """Circular convolution implementation."""
    result = torch.zeros(wg.size())
    for i in range(batch_size):
        w = wg[i]
        s = sg[i]
        assert s.size(0) == 3
        t = torch.cat([w[-1:], w, w[:1]])
        result[i] = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    # if CUDA:
    # result = result.cuda()
    return result


class Memory(nn.Module):
    def __init__(self, N, M, init_mode='const'):
        super(Memory, self).__init__()

        self.M = M
        self.N = N

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(N, M))
        if init_mode == 'const':
            nn.init.constant_(self.mem_bias, 1e-6)
        elif init_mode == 'random':
            std_dev = 1 / np.sqrt(N + M)
            nn.init.uniform_(self.mem_bias, -std_dev, std_dev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def read(self, address):
        """
        :param address: Batched Tensor with Size of batch_size * N, contain value between 0 and 1 with sum equals to 1
        :return: Torch batched tensor with Size of batch_size * M, produce by sum over weighted elements of Memory
        """
        return address.unsqueeze(1).matmul(self.memory).squeeze(1)

    def write(self, address, erase_vector, add_vector):
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        # if CUDA:
        # self.memory = self.memory.cuda()
        erase = torch.matmul(address.unsqueeze(-1), erase_vector.unsqueeze(1))
        add = torch.matmul(address.unsqueeze(-1), add_vector.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, key_vector, key_strength, gate, shift, sharpen, last_address):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param key_vector: The key vector.
        :param key_strength: The key strength (focus).
        :param gate: Scalar interpolation gate (with previous weighting).
        :param shift: Shift weighting.
        :param sharpen: Sharpen weighting scalar.
        :param last_address: The weighting produced in the previous time step.
        """
        wc = F.softmax(key_strength * F.cosine_similarity(key_vector.unsqueeze(1), self.memory, dim=2), dim=1)
        wg = (gate * wc) + (1 - gate) * last_address
        ws = _convolve(wg, shift, self.batch_size)
        ws = (ws ** sharpen)
        wt = torch.div(ws, torch.sum(ws, dim=1).view(-1, 1) + 1e-16)

        return wt

    def size(self):
        return self.N, self.M


class LSTMController(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_layers, batch_size):
        super(LSTMController, self).__init__()

        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.num_outputs = num_outputs

        self.lstm = nn.LSTM(num_inputs, num_outputs, num_layers)

        self.lstm_h = nn.Parameter(torch.randn(num_layers, batch_size, num_outputs) * 0.05)  # .cuda()  # Why 0.05??
        self.lstm_c = nn.Parameter(torch.randn(num_layers, batch_size, num_outputs) * 0.05)  # .cuda()

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        return [self.lstm_h, self.lstm_c]

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs + self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state


"""## Heads

### 1. Base Head
"""


class BaseHead(nn.Module):
    def __init__(self, memory, controller):
        super(BaseHead, self).__init__()

        self.memory = memory
        _, self.ctrl_size = controller.size()
        self.M = memory.M
        self.N = memory.N

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, k, B, g, s, L, w_prev):
        # Handle Activations
        k = F.tanh(k)
        B = F.softplus(B)
        g = torch.sigmoid(g)
        s = F.softmax(s, dim=1)
        L = 1 + F.softplus(L)

        w = self.memory.address(k, B, g, s, L, w_prev)

        return w


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


"""### 2. Read Head"""


class ReadHead(BaseHead):
    def __init__(self, memory, controller):
        super(ReadHead, self).__init__(memory, controller)

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_vector = [self.M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(self.ctrl_size, sum(self.read_vector))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        w_perv = torch.zeros(batch_size, self.N)
        # if CUDA:
        # w_perv = w_perv.cuda()
        return w_perv

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, input, last_w):
        out = self.fc_read(input)
        K, B, G, S, L = _split_cols(out, self.read_vector)
        w = self._address_memory(K, B, G, S, L, last_w)
        r = self.memory.read(w)
        return r, w


"""### 3. Write Head"""


class WriteHead(BaseHead):
    def __init__(self, memory, controller):
        super(WriteHead, self).__init__(memory, controller)

        #                     K, B, G, S, L, add, erase
        self.write_vector = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(self.ctrl_size, sum(self.write_vector))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        w_perv = torch.zeros(batch_size, self.N)
        # if CUDA:
        # w_perv = w_perv.cuda()
        return w_perv

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, input, last_w):
        out = self.fc_write(input)
        K, B, G, S, L, A, E = _split_cols(out, self.write_vector)
        w = self._address_memory(K, B, G, S, L, last_w)
        self.memory.write(w, torch.sigmoid(E), F.tanh(A))
        return w


"""## DataPath"""


class NTMCell(nn.Module):
    """A DataPath for NTM."""

    def __init__(self, num_inputs, num_outputs, controller, memory, heads):
        """Initialize the DataPath.
        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`LSTMController`
        :param memory: :class:`Memory`
        :param heads: list of :class:`ReadHead` or :class:`WriteHead`
        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(NTMCell, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads

        self.N, self.M = memory.size()
        _, self.controller_size = controller.size()

        # Initialize the initial previous read values to random biases
        self.num_read_heads = 0
        self.init_r = []
        for head in self.heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "heads list must contain at least a single read head"

        # Initialize a fully connected layer to produce the actual output:
        #   [controller_output; previous_reads ] -> output
        self.fc = nn.Linear(self.controller_size + self.num_read_heads * self.M, num_outputs)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]

        return init_r, controller_state, heads_state

    def reset_parameters(self):
        # Initialize the linear layer
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, x, prev_state):
        """DataPath forward function.
        :param x: input vector (batch_size x num_inputs)
        :param prev_state: The previous state of the DataPath
        """
        # Unpack the previous state
        prev_reads, prev_controller_state, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        # if CUDA:
        # x = x.cuda()
        # for i in range(len(prev_reads)):
        #     prev_reads[i] = prev_reads[i].cuda()
        #
        # for i in range(len(prev_controller_state)):
        #     prev_controller_state[i] = prev_controller_state[i].cuda()

        # x = x.flatten()
        # print('x', x.shape)
        # print(len(prev_reads), prev_reads[0].shape)
        inp = torch.cat([x] + prev_reads, dim=1)
        # if CUDA:
        # inp = inp.cuda()
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                r, head_state = head(controller_outp, prev_head_state)
                # if CUDA:
                # r = r.cuda()
                # head_state = head_state.cuda()
                reads += [r]
            else:
                head_state = head(controller_outp, prev_head_state)
                # if CUDA:
                # head_state = head_state.cuda()
            heads_states += [head_state]

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = torch.sigmoid(self.fc(inp2))
        # o = torch.sigmoid(self.fc(controller_outp))

        # Pack the current state
        state = (reads, controller_state, heads_states)

        return o, state


"""NTM"""


class NTM(nn.Module):

    def __init__(self, num_inputs, num_outputs, controller_size, controller_layers, num_read_heads, num_write_heads, N,
                 M, batch_size):
        """Initialize an NTM.
        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(NTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.N = N
        self.M = M

        # Create the NTM components
        memory = Memory(N, M)
        controller = LSTMController(num_inputs + (M * num_read_heads), controller_size, controller_layers, batch_size)
        heads = nn.ModuleList([ReadHead(memory, controller) for _ in range(num_read_heads)])
        heads += [WriteHead(memory, controller) for _ in range(num_write_heads)]

        self.data_path = NTMCell(num_inputs, num_outputs, controller, memory, heads)
        self.memory = memory

        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.data_path.create_new_state(batch_size)

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.data_path.create_new_state(batch_size)

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs)
        # if CUDA:
        # x = x.cuda()
        o, self.previous_state = self.data_path(x, self.previous_state)
        return o, self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)

        return num_params


class NTMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NTMAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.ntm = NTM(args.rnn_hidden_dim, args.n_actions, 100, 1, 1, 1, 128, 20, 160)
        self.fc2 = nn.Linear(input_shape, args.n_actions)

        # self.fc1.cuda()
        # self.fc2.cuda()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if inputs.shape[0] != 160:
            return torch.zeros(5, 11), torch.zeros(hidden_state.shape)
        x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        q, h = self.ntm(x)
        # q = self.fc2(h)
        return q, h[1]


if __name__ == '__main__':
    a = NTM(10, 10, 100, 1, 1, 1, 128, 20, 6)
    b = copy.deepcopy(a)
