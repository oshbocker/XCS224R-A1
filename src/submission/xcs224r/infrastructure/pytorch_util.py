'''
TO EDIT: Utilities for handling PyTorch

Functions to edit:
    1. build_mlp (line 26) 
'''


from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

device = None

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        Arguments:
            n_layers: number of total layers, computed as number of hidden layers + 1
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        Returns:
            MLP (nn.Module)


        TODO:
            Build a feed-forward network (multi-layer perceptron, or mlp) that maps
            input_size-dimensional vectors to output_size-dimensional vectors.
            It should have 'n_layers - 1' hidden layers, each of 'size' units and followed
            by a nonlinearity function (`activation`). The final layer should be linear followed
            by a nonlinearity function (`output_activation`).

            Recall a hidden layer is a layer that occurs between the input and output
            layers of the network.

            As part of your implementation please make use of the following Pytorch
            functionalities:
            nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
            nn.Sequential (https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)

        Hint:
            It is possible to create a list of nn.Modules and unpack these into nn.Sequential.
            For example:
                modules = []
                modules.append(nn.Linear(10, 10))
                modules.append(nn.Linear(10, 10))
                model = nn.Sequential(*modules)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    mlp = None

    # *** START CODE HERE ***
    input_layer = nn.Linear(input_size, size)
    modules = []
    modules.append(input_layer)
    modules.append(activation)
    for i in range(n_layers - 1):
        modules.append(nn.Linear(size, size))
        modules.append(activation)
    output_layer = nn.Linear(size, output_size)
    modules.append(output_layer)
    modules.append(output_activation)
    mlp = nn.Sequential(*modules)
    # *** END CODE HERE ***

    return mlp


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and use_gpu:
        device = torch.device("mps")
        print("PyTorch detects an Apple GPU: running on MPS")
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
