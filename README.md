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


**1.5** Reward and Return

Reward function is defined as the reward obtained by the agent transfer from s_t to s_{t+1} when taking action a_t.

![Eqn](https://microsoft.codecogs.com/svg.latex?r_t%20%3D%20R%28s_t%2C%20a_t%2C%20s_%7Bt%2B1%7D%29)
<!--$$r_t = R(s_t, a_t, s_{t+1})$$-->


The target of reinforcement learning is maximizating the reward function over a trajectiory, and there are several different ways to calucate the overall reward summations.

1. Finite-horizon undiscounted return (may diverge over infinite sum)

![Eqn](https://microsoft.codecogs.com/svg.latex?R%28%5Ctau%29%20%3D%20%5Csum%5ET_%7Bt%3D0%7D%20r_t)
<!--$$R(\tau) = \sum^T_{t=0} r_t$$-->


2. Infinite-horizon discounted return

![Eqn](https://microsoft.codecogs.com/svg.latex?%20R%28%5Ctau%29%20%3D%20%5Csum%5E%7B%5Cinfty%7D_%7Bt%3D0%7D%20%5Cgamma%5Et%20r_t%20)
<!--$$ R(\tau) = \sum^{\infty}_{t=0} \gamma^t r_t $$-->


**1.6** RL Problem
The RL problem: select a policy which maximizes expected return when the agent acts according to it. 
The probablity of a trajectory given a policy can be described as follows,

![Eqn](https://microsoft.codecogs.com/svg.latex?P%28%5Ctau%7C%5Cpi%29%20%3D%20%5Crho_0%28s_0%29%5Cprod%5E%7BT-1%7D_%7Bt%3D0%7DP%28s_%7Bt%2B1%7D%7Cs_t%2C%20a_t%29%5Cpi%28a_t%7Cs_t%29)
<!--$$P(\tau|\pi) = \rho_0(s_0)\prod^{T-1}_{t=0}P(s_{t+1}|s_t, a_t)\pi(a_t|s_t)$$-->


The expected return can be described as follows,

![Eqn](https://microsoft.codecogs.com/svg.latex?J%28%5Cpi%29%3D%5Cint_%5Ctau%20P%28%5Ctau%7C%5Cpi%29R%28%5Ctau%29%20%3D%20%5Cunderset%7B%5Ctau%20%5Csim%20%5Cpi%7D%7B%5Cmathop%7B%5Cmathbb%7BE%7D%7D%7D%20%5BR%28%5Ctau%29%5D%20)
<!--$$J(\pi)=\int_\tau P(\tau|\pi)R(\tau) = \underset{\tau \sim \pi}{\mathop{\mathbb{E}}} [R(\tau)] $$-->


The optimal policy can be described as follows,

![Eqn](https://microsoft.codecogs.com/svg.latex?%5Cpi%5E%2A%20%3D%20%5Cunderset%7B%5Cpi%7D%7Bargmax%7DJ%28%5Cpi%29)
<!--$$\pi^* = \underset{\pi}{argmax}J(\pi)$$-->


**1.7** Value functions
Many kinds of value functions can be used to estimate the value of the state. Followings are a list of thest value functions.

1. **On-Policy Value Function**

![Eqn](https://microsoft.codecogs.com/svg.latex?%20V%5E%7B%5Cpi%7D%28s%29%20%3D%20%20%5Cunderset%7B%5Ctau%20%5Csim%20%5Cpi%7D%7B%5Cmathop%7B%5Cmathbb%7BE%7D%7D%7D%20%5BR%28%5Ctau%29%20%7C%20s_0%20%3D%20s%5D%20)
<!--$$ V^{\pi}(s) =  \underset{\tau \sim \pi}{\mathop{\mathbb{E}}} [R(\tau) | s_0 = s] $$-->


2. **On-Policy Action-Value Function**
![Eqn](https://microsoft.codecogs.com/svg.latex?%20Q%5E%7B%5Cpi%7D%28s%2C%20a%29%20%3D%20%5Cunderset%7B%5Ctau%20%5Csim%20%5Cpi%7D%7B%5Cmathop%7B%5Cmathbb%7BE%7D%7D%7D%20%5BR%28%5Ctau%29%20%7C%20s_0%20%3D%20s%2C%20a_0%20%3D%20a%5D%20)
<!--$$ Q^{\pi}(s, a) = \underset{\tau \sim \pi}{\mathop{\mathbb{E}}} [R(\tau) | s_0 = s, a_0 = a] $$-->


3. **Optimal Value Function**
![Eqn](https://microsoft.codecogs.com/svg.latex?%20V%5E%2A%28s%29%20%3D%20%20%5Cunderset%7B%5Cpi%7D%7Bmax%7D%5Cunderset%7B%5Ctau%20%5Csim%20%5Cpi%7D%7B%5Cmathop%7B%5Cmathbb%7BE%7D%7D%7D%20%5BR%28%5Ctau%29%20%7C%20s_0%20%3D%20s%5D)
<!--$$ V^*(s) =  \underset{\pi}{max}\underset{\tau \sim \pi}{\mathop{\mathbb{E}}} [R(\tau) | s_0 = s]$$-->


4. **Optimal Action-Value Function**
![Eqn](https://microsoft.codecogs.com/svg.latex?%20Q%5E%2A%28s%2C%20a%29%20%3D%20%5Cunderset%7B%5Cpi%7D%7Bmax%7D%5Cunderset%7B%5Ctau%20%5Csim%20%5Cpi%7D%7B%5Cmathop%7B%5Cmathbb%7BE%7D%7D%7D%20%5BR%28%5Ctau%29%20%7C%20s_0%20%3D%20s%2C%20a_0%20%3D%20a%5D%20)
<!--$$ Q^*(s, a) = \underset{\pi}{max}\underset{\tau \sim \pi}{\mathop{\mathbb{E}}} [R(\tau) | s_0 = s, a_0 = a] $$-->


There are two obvious relations,
![Eqn](https://microsoft.codecogs.com/svg.latex?%20V%5E%7B%5Cpi%7D%28s%29%20%3D%20%5Cunderset%7Ba%20%5Csim%20%5Cpi%7D%7B%5Cmathop%7B%5Cmathbb%7BE%7D%7D%20%7D%20%5BQ%5E%7B%5Cpi%7D%28s%2C%20a%29%5D%20)
<!--$$ V^{\pi}(s) = \underset{a \sim \pi}{\mathop{\mathbb{E}} } [Q^{\pi}(s, a)] $$-->


![Eqn](https://microsoft.codecogs.com/svg.latex?%20%20V%5E%2A%28s%29%20%3D%20%5Cunderset%7Ba%7D%7Bmax%7D%20%20Q%5E%2A%28s%2C%20a%29%20)
<!--$$  V^*(s) = \underset{a}{max}  Q^*(s, a) $$-->


**1.8** The Optimal Action-Value Function and the Optimal Action
If we know the Optimal Action-Value Function (which means the value after taking a randomly choosed action, then always choose the optimal policy), we can get the optimal action according to the following equation.
![Eqn](https://microsoft.codecogs.com/svg.latex?a%5E%2A%28s%29%20%3D%20%5Cunderset%7Ba%7D%7Bargmax%7DQ%5E%2A%28s%2C%20a%29)
<!--$$a^*(s) = \underset{a}{argmax}Q^*(s, a)$$-->


**1.9** Bellman Equations
