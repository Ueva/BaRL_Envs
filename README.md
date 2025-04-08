# SimpleEnvs

A collection of Reinforcement Learning (RL) environments which I have implemented as part of my research. Hopefully this collection will grow over time, and prove useful to others as well.

All of these environments use an interaction API similar to that used by [OpenAI Gym](https://github.com/openai/gym), however we do not implement Gym's `Env` class directly. Instead, we use the [SimpleOptions](https://github.com/Ueva/BaRL-SimpleOptions/blob/master/simpleoptions/environment.py) `BaseEnvironment` interface.

Many of the smaller, discrete environments (e.g. Taxi, Hanoi, Rooms) are designed with graph-based reinforcement learning methods in mind. They naturally support building state transition graphs out-of-the-box using the built-in `generate_interaction_graph` method, and one-step lookahead is available using the `get_successors` method.

The easiest way to install this package is to simply run `pip install simpleenvs`. Alternatively, you can install from source: simply download this repository and run the command `pip install .` in the root directory.

The code in this repository is well commented, but additional examples and documentation are coming soon!
