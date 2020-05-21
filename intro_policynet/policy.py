#! /usr/bin/env python

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import unittest

def catetgorical_wrapper(func):
    def wrapped(*args, **kwargs):
        ret = func(*args, **kwargs)
        sampler = Categorical(logits=ret)
        return sampler
    return wrapped

def gaussian_wrapper(func):
    def wrapped(*args, **kwargs):
        ret = func(*args, **kwargs)
        mu, logs = torch.chunk(ret, 2)
        s = logs.exp()
        sampler = Normal(loc=mu, scale=s)
        return sampler
    return wrapped

class Policy(nn.Module):
    def __init__(self, feat_dim, act_dim):
        """ This class defines the policy network
            feat_dim: size of input features
            act_dim: size of possible actions
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.act_dim = act_dim

        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanhshrink(),
            nn.Linear(64, act_dim)
        )

    def forward(self, feat):
        return self.net(feat)


class TestPolicy(unittest.TestCase):
    def test_policynet(self):
        model = Policy(6, 4)
        feat = torch.randn(4, 6)
        print("Model Output:")
        print(model(feat))

    def test_categorical_policy(self):
        model = Policy(6, 4)
        feat = torch.randn(4, 6)
        model.forward = catetgorical_wrapper(model.forward) 
        sampler = model(feat)
        s = sampler.sample()
        p = sampler.log_prob(s)
        print("Categorical Output:")
        print(s)
        print("Categorical log probs:")
        print(p)

    def test_gaussian_policy(self):
        model = Policy(6, 4)
        feat = torch.randn(4, 6)
        model.forward = gaussian_wrapper(model.forward) 
        sampler = model(feat)
        print("Gaussian output (undifferentiable)")
        s = sampler.sample()
        print(s)
        print("Gaussian output (differentiable)")
        s = sampler.rsample()
        print(s)
        print("Log probs:")
        print(sampler.log_prob(s))
