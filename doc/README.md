# Notes on Reinforcement Learning Spinning Up

This repo serves as a course note on [OpenAI Sppining Up](https://spinningup.openai.com/en/latest/).
The deep learning framework used here is [PyTorch 1.5](https://https://pytorch.org/).

## 1. [Policy Network Basis](./intro_policynet/policy.py)

**1.1** Sample Categorical Distribution

Sampling categorical distribution follows the following flow,
Neural Network -> Logits -> Sample -> Logprob
By this way we can simply have the final sampled entries and log probs.

**1.2** Sample Gaussian Distribution

Basically there are two ways to sample Gaussian distribution,

1. Using the $\mu$ produced by neural network, and $log \sigma$ as a prameter (VPG, TRPO and PPO way)
2. Using neural network to generate both mu and sigma

*Notice: $\mu$ and $log \sigma$ has range $(-\infty, \infty)$*

**1.3** Reparameterization Trick

For Gaussian distribution, in order to make distribution differntiable, we need to use some method called reparameterization trick. So we can simply generate $\mu$ and $\sigma$ with NN models, then, using this trick (with generated standard Gaussian distribution numbers), to obtain the final differeitiable output.

$$\mathbf{X} \sim N(\mu, \sigma) \to \mathbf{X} \sim \mu + \sigma \cdot N(0, 1) $$

**1.4** Trajectories

A trajectory $\tau$ is defined as a squence of state $s_t$ and action $a_t$.

$$ \tau = (s_0, a_0, s_1, a_1, ...)$$

The first state is sampled from a distribution.

$$ s_0 \sim \rho_0(\cdot)$$

Deterministic policy is defined as follows,

$$s_{t+1} = f(s_t + a_t)$$

Stochastic policy is defined as follows,

$$s_{t+1} \sim P(\cdot | s_t, a_t)$$

**1.5** Reward and Return

Reward function is defined as the reward obtained by the agent transfer from s_t to s_{t+1} when taking action a_t.

$$r_t = R(s_t, a_t, s_{t+1})$$

The target of reinforcement learning is maximizating the reward function over a trajectiory, and there are several different ways to calucate the overall reward summations.

1. Finite-horizon undiscounted return (may diverge over infinite sum)

$$R(\tau) = \sum^T_{t=0} r_t$$

2. Infinite-horizon discounted return

$$ R(\tau) = \sum^{\infty}_{t=0} \gamma^t r_t $$

**1.6** RL Problem

The RL problem: select a policy which maximizes expected return when the agent acts according to it. 
The probablity of a trajectory given a policy can be described as follows,

$$P(\tau|\pi) = \rho_0(s_0)\prod^{T-1}_{t=0}P(s_{t+1}|s_t, a_t)\pi(a_t|s_t)$$

The expected return can be described as follows,

$$J(\pi)=\int_\tau P(\tau|\pi)R(\tau) = \underset{\tau \sim \pi}{\mathop{\mathbb{E}}} [R(\tau)] $$

The optimal policy can be described as follows,

$$\pi^* = \underset{\pi}{argmax}J(\pi)$$

**1.7** Value functions

Many kinds of value functions can be used to estimate the value of the state. Followings are a list of thest value functions.

1. **On-Policy Value Function**

$$ V^{\pi}(s) =  \underset{\tau \sim \pi}{\mathop{\mathbb{E}}} [R(\tau) | s_0 = s] $$

2. **On-Policy Action-Value Function**

$$ Q^{\pi}(s, a) = \underset{\tau \sim \pi}{\mathop{\mathbb{E}}} [R(\tau) | s_0 = s, a_0 = a] $$

3. **Optimal Value Function**
 
$$ V^*(s) =  \underset{\pi}{max}\underset{\tau \sim \pi}{\mathop{\mathbb{E}}} [R(\tau) | s_0 = s]$$

4. **Optimal Action-Value Function**

$$ Q^*(s, a) = \underset{\pi}{max}\underset{\tau \sim \pi}{\mathop{\mathbb{E}}} [R(\tau) | s_0 = s, a_0 = a] $$

There are two obvious relations,
$$ V^{\pi}(s) = \underset{a \sim \pi}{\mathop{\mathbb{E}} } [Q^{\pi}(s, a)] $$

$$  V^*(s) = \underset{a}{max}  Q^*(s, a) $$

**1.8** The Optimal Action-Value Function and the Optimal Action

If we know the Optimal Action-Value Function (which means the value after taking a randomly choosed action, then always choose the optimal policy), we can get the optimal action according to the following equation.
$$a^*(s) = \underset{a}{argmax}Q^*(s, a)$$

**1.9** Bellman Equations

The four formulas in 1.7 can be express with Bellman equations, therefore we have:

Bellman equation for on-policy value function:

$$ V^\pi(s) = \underset{s' \sim P}{\underset{a \sim \pi}{\mathop{\mathbb{E}}}} [r(s, a) + \gamma V^\pi(s')] $$

$$Q^\pi(s, a) = \underset{s' \sim P}{\mathop{\mathbb{E}}} [r(s, a) + \gamma \underset{a' \sim \pi}{\mathop{\mathbb{E}}}[Q^\pi(s', a')]$$

Bellman equation for optimal value function:

$$ V^*(s) = \underset{a}{max}\underset{s' \sim p}{\mathop{\mathbb{E}}} [r(s, a) + \gamma V^*(s')] $$

$$Q^*(s, a) = \underset{s' \sim P}{\mathop{\mathbb{E}}} [r(s, a) + \gamma \underset{a'}{max}[Q^*(s', a')]$$

**1.10** Advantage Funtions

By using advantage functions, we can know how much a policy gains comparing to the average. The equation is defined by the followings.

$$ A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

##  2. Kinds of RL Algorithms

**2.1** Category of RL Algorithms

This figure is from OpenAI's spinning up RL. As the figure indicates, basically the RL Algorithms can be classifed as Model-Free RL and Model-Based RL.

![Classifcation Image](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

**2.2** Model-Free RL

