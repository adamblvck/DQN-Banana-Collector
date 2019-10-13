# DQN-Banana-Collector

To train an agent based on feedback from its environment, it's self-evident that a reinforcement learning algorithm would serve this problem well.

In Udacity's [Deep Reinforcement Learning](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) course we went through classical Q-learning algorithms like SARSA, learned the limitations of that algoriths, and have seen the *deep* powers of Deep Q-Networks (abbreviated as DQNs).

Seeing the efficiency at which DQNs can solve agents in [OpenAI Gym](https://openai.com/), it only made sense to try the DQN-algorithm for our Banana-Collector.

The Banana-Collector environment provides us with a state with 37 dimensions, including ray-based detections, velocity and others. The Banana-Collector is also reduced to only 4 actions.

The implementation makes use of `Python 3` and `PyTorch`. Let's see how we've chosen the learning algorithm.

## Learning Algorithm

A Deep Q-Network essentially tries to find an optimal strategy by constructing a Q-table using several layers of feedforward neural networks.

![](https://miro.medium.com/proxy/1*DW0Ccmj1hZ0OvSXi7Kz5MQ.jpeg)

A feedforward neural network consists of an input layer, hidden layers and an output layer.

Since we're trying to approximate a Q-table with these components, it makes sense to set the input layer to 37 neurons (this is the state- space containing 37 dimensions), while choosing 4 neurons in the output layer (our agent can only perform 4 actions).

The hidden-layers in a neural network are something to be chosen experimentally, alongside many other hyperparameters.

We've also implemented `Fixed Q-targets` with a `Replay Buffer`, to avoid the agent getting stuck in loops. The neural network is optimized with an `ADAM` optimizer.

Neural Network Architecture (in number of neurons per layer):
- Input: 37 neurons
- fc1: 74 neurons
- fc2: 74 neurons
- Output: 4 neurons

The hyperparameters of the learning algorithm are as follows:
- Gamma: 0.99 - The discount rate of the algorithm
- Learning Rate: 5e-4 - The learning rate of the ADAM-optimizer.
- Update Every: 4 - The number of steps before updating fixed targets
- Tau: 1e-3 - Soft updating rate of target params for fixed Q-targets

## Plot of Rewards

The project specification specifies that the environment is considered to be solved when the average score in 100 episodes is over 13. We've chosen the score to be 16.

On episode `835`, the DQN-agent managed to solve the environment with a score of exactly 16. The training output looks as follows:

~~~
Episode 100	Average Score: 0.64
Episode 200	Average Score: 3.77
Episode 300	Average Score: 7.02
Episode 400	Average Score: 9.84
Episode 500	Average Score: 13.05
Episode 600	Average Score: 14.63
Episode 700	Average Score: 14.81
Episode 800	Average Score: 15.00
Episode 900	Average Score: 15.46
Episode 935	Average Score: 16.00
Environment solved in 835 episodes!	Average Score: 16.00
~~~

Below a plot with the scores as function in episodes, for a visual representation of the learning process.

![](media/dqn_agent_episode_x_scores/png)

To the the code, please take a look at [the python notebook](Navigation.ipynb).

## Ideas for Future Work

The agent is collecting yelllow bananas and avoiding blue bananas pretty well. I've noticed that sometimes the agent gets stuck moving `LEFT` and `RIGHT`, this is when he's only seeing blue bananas in the environment. This could be combatted by increasing the `UPDATE EVERY` hyper-parameter to 8 or 16.

The learning algorithm could be expanded with other algorithms, including `Double DQN`, `Dueling DQN` and `prioritized experience replay`. I'm not sure how far implementing these would increase the score. To see the effect of these algorithms in action, I'd propose to rebuild the environment with the following adjustments:
- Add a turning-speed to the agent. Pressing left or right accelerates the turn speed left or right, with friction factor that steadily diminishes the turning speed to zero. This would make it harder for the agent to move around, and thus it would be more important for `double DQN` or  `prioritized experience replay` to be implemented.
- Add obstacles, valleys, and bridges to the game, similar to the game [Quake](https://youtu.be/ZHT2TgMX7Rg). In a more complex environment, implementing above-mentioned fine-tunings (and more) would certainly be required to consider that environment solved.