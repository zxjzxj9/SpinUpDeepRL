# Notes on Reinforcement Learning Spinning Up

This repo serves as a course note on [OpenAI Sppining Up](https://spinningup.openai.com/en/latest/).
The deep learning framework used here is [PyTorch 1.5](https://https://pytorch.org/).

## 1. [Policy Network Basis](./intro_policynet/policy.py)

**1.1** Sample categorical distribution

Sampling categorical distribution follows the following flow,
Neural Network -> Logits -> Sample -> Logprob
By this way we can simply have the final sampled entries and log probs.

**1.2** Sample Gaussian distribution

Basically there are two ways to sample Gaussian distribution,

1. Using the $\mu$ produced by neural network, and $log \sigma$ as a prameter (VPG, TRPO and PPO way)
2. Using neural network to generate both mu and sigma

*Notice: $\mu$ and $log \sigma$ has range $(-\infty, \infty)$*

**1.3** Reparameterization trick

For Gaussian distribution, in order to make distribution differntiable, we need to use some method called reparameterization trick. So we can simply generate $\mu$ and $\sigma$ with NN models, then, using this trick (with generated standard Gaussian distribution numbers), to obtain the final differeitiable output.

![Eqn](https://microsoft.codecogs.com/svg.latex?%5Cmathbf%7BX%7D%20%5Csim%20N%28%5Cmu%2C%20%5Csigma%29%20%5Cto%20%5Cmathbf%7BX%7D%20%5Csim%20%5Cmu%20%2B%20%5Csigma%20%5Ccdot%20N%280%2C%201%29%20)
<!--$$\mathbf{X} \sim N(\mu, \sigma) \to \mathbf{X} \sim \mu + \sigma \cdot N(0, 1) $$-->


**1.4** Trajectorie

A trajectory $\tau$ is defined as a squence of state $s_t$ and action $a_t$.

![Eqn](https://microsoft.codecogs.com/svg.latex?%20%5Ctau%20%3D%20%28s_0%2C%20a_0%2C%20s_1%2C%20a_1%2C%20...%29)
<!--$$ \tau = (s_0, a_0, s_1, a_1, ...)$$-->


The first state is sampled from a distribution.

![Eqn](https://microsoft.codecogs.com/svg.latex?%20s_0%20%5Csim%20%5Crho_0%28%5Ccdot%29)
<!--$$ s_0 \sim \rho_0(\cdot)$$-->


Deterministic policy is defined as follows,

![Eqn](https://microsoft.codecogs.com/svg.latex?s_%7Bt%2B1%7D%20%3D%20f%28s_t%20%2B%20a_t%29)
<!--$$s_{t+1} = f(s_t + a_t)$$-->


Stochastic policy is defined as follows,

![Eqn](https://microsoft.codecogs.com/svg.latex?s_%7Bt%2B1%7D%20%5Csim%20P%28%5Ccdot%20%7C%20s_t%2C%20a_t%29)
<!--$$s_{t+1} \sim P(\cdot | s_t, a_t)$$-->

