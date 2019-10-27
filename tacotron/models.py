import torch
from torch import nn

import tacotron.modules

class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()
