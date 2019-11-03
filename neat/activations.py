import torch
import torch.nn.functional as F


def sigmoid(x):
    return torch.sigmoid(x)


def tanh(x):
    return torch.tanh(x)

######### Added by Amin ######
def ReLU(x):
    m = torch.nn.ReLU()
    return m(x)


class Activations:

    def __init__(self):
        self.functions = dict(
            sigmoid=sigmoid,
            tanh=tanh,
            ######### Added by Amin ######
            ReLU=ReLU
        )

    def get(self, func_name):
        return self.functions.get(func_name, None)
