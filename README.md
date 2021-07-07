# BaRL_Envs
A collection of Reinforcement Learning (RL) environments which I have implemented as part of my research. Hopefully this collection will grow over time, and prove useful to others as well.

All of these environments use an interaction API similar to that used by [OpenAI Gym](https://github.com/openai/gym), however we do not implement Gym's `Env` class directly.

Further documentation coming soon!

Many of the smaller, discrete environments (e.g. Taxi, Hanoi, Rooms) are designed to be friendly for use with graph-based methods. If you wish to construct a state-transition graph for an environment, you can do so like [this](https://gist.github.com/Ueva/aadbb0b396466ad2a16dc629ba924b45).
