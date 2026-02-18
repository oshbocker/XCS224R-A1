"""
TO EDIT: Defines a pytorch policy as the agent's actor.

Functions to edit:
    1. get_action (line 111)
    2. forward (line 126)
    3. update (line 141)
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from ..infrastructure import pytorch_util as ptu
from ..policies.base_policy import BasePolicy


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to actions

    Attributes
    ----------
    logits_na: nn.Sequential
        A neural network that outputs dicrete actions
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    get_action:
        Calls the actor forward function
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # Initialize variables for environment (action/observation dimension, number of layers, etc.)
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # NOTE: This works for a continuous action space. All our environments use a continuous action space.
        self.logits_na = None
        self.mean_net = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    ##################################

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        :param obs: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        # *** START CODE HERE ***
        observation = ptu.from_numpy(observation.astype(np.float32))
        with torch.no_grad():
            action = ptu.to_numpy(self.forward(observation).sample())
        return action
        # *** END CODE HERE ***

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action distribution (torch.distributions.Distribution): a pytorch distribution representing
                a diagonal Gaussian distribution whose mean (loc) is computed by
                self.mean_net and standard deviation (scale) is torch.exp(self.logstd)

        Note:
            PyTorch doesn't have a diagonal Gaussian built in, but you can
            fashion one out of
            (a) torch.distributions.MultivariateNormal
            or
            (b) a combination of torch.distributions.Normal
                             and torch.distributions.Independent

            Please consult the following documentation for further details on
            the use of probability distributions in Pytorch:
            https://pytorch.org/docs/stable/distributions.html
        """

        # *** START CODE HERE ***
        action_mean = self.mean_net.forward(observation)
        action_std = torch.exp(self.logstd)
        dist = distributions.multivariate_normal.MultivariateNormal(
            loc=action_mean,
            scale_tril=torch.diag(action_std)
        )
        return dist
        # *** END CODE HERE ***

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss. Recall that to update the policy
        # you need to backpropagate the gradient and step the optimizer.

        # *** START CODE HERE ***
        criterion = torch.nn.CrossEntropyLoss()
        observations = ptu.from_numpy(observations.astype(np.float32))
        actions = ptu.from_numpy(actions.astype(np.float32))
        output_actions = self.mean_net.forward(observations).rsample()
        loss = criterion(output_actions, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return {"Training Loss": ptu.to_numpy(loss)}
        # *** END CODE HERE ***
