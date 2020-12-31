REGISTRY = {}

from .rnn_agent import RNNAgent
from .noisy_agent import NoisyRNNAgent
from .ntm_agent import NTMAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["ntm"] = NTMAgent
REGISTRY["noisy-rnn"] = NoisyRNNAgent
