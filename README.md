# gymtime
Deep RL algorithms for OpenAI gym's [environments](https://gym.openai.com/envs)


See Heerad's submissions [here](https://gym.openai.com/users/heerad)

Implemented:
* Actor-critic with per-step updates using eligibiilty traces
* Deep Q-learning (DQN) with experience replay to improve sample efficiency
* [DDPG](https://arxiv.org/pdf/1509.02971.pdf) for continuous action spaces
* [UCB exploration](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/XX.pdf) based on Hoeffding's inequality as an alternative to epsilon-greedy exploration for DQN
* [Double Q-learning](https://arxiv.org/abs/1509.06461) for eliminating maximization bias from applying function approximators to Q-learning
* [Prioritized experience replay](https://arxiv.org/pdf/1511.05952.pdf) for DQN
* Slowly-updating target network (used in computing TD error) for stability
* Norm clipping for stability

TODO:
* Atari environments via convnets
* [PPO](https://arxiv.org/abs/1707.06347)