# Community Detection with Reinforcement Learning

This project applies reinforcement learning to the task of modularity maximization for community detection on graphs.

Each episode in the environment follows the following steps:
<p align="left">
    <img src="assets/environment.png" width="300" />
</p>

An adaptation of Deep Q-Learning is used to optimize the modularity-based reward signal. On synthetic datasets, the agent can almost perfectly recover the community structure used to generate the graphs. Below is the reward trace and the modularity trace over the course of training on a synthetic graph:
<p align="center">
  <img src="assets/synth_reward.png" width="49%">
  <img src="assets/synth_modularity.png" width="49%">
</p>

See the [write-up](final-report.pdf) for full technical details.
