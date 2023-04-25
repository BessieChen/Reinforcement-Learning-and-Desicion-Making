Three experiments are conducted in this project based on reinforcement learning:


### Experiment 1
This experiment demonstrates how to apply the Deep Q-Network (DQN) to the Lunar Lander environment built by OpenAI gym, aiming to achieve an average score of over 200 for the last 100 experiences within the pre-specified maximum episode time. The experiment also compared the performance of DQN and Double Deep Q-Learning (DDQN) algorithms, indicating that DDQN effectively alleviates the overestimation problem, leading to better performance. Adjusting hyperparameters such as α and ε is essential for DQN to play its role.
![输入图片说明](02%20Lunar%20Lander%20Solver%20Based%20on%20DQN%20and%20DDQN%20Methods/Fig%202&3..png)
![输入图片说明](02%20Lunar%20Lander%20Solver%20Based%20on%20DQN%20and%20DDQN%20Methods/Fig%204&6..png)
![输入图片说明](02%20Lunar%20Lander%20Solver%20Based%20on%20DQN%20and%20DDQN%20Methods/Fig%205&7..png)

### Experiment 2
This experiment explores the application of multi-agent Q-learning algorithms in the reinforcement learning field, specifically implementing classic Q-learning with ε-greedy and three other variants of correlated Q-learning algorithms in a zero-sum soccer game. The experiment results showed that Foe-Q, uCE-Q, and Friend-Q converged, except for classic Q-learning. However, Friend-Q converged to an irrationally optimistic equilibrium, while Foe-Q and CE-Q shared identical results due to the zero-sum nature, having the same probability distribution in the joint action space. In contrast, Q-learning did not converge as players lacked information about their opponents and acted independently.
![输入图片说明](03%20Multi-Agent%20Zero-Sum%20Soccer%20Game%20Experiment/result_figures/q.png)
![输入图片说明](03%20Multi-Agent%20Zero-Sum%20Soccer%20Game%20Experiment/result_figures/friend.png)
![输入图片说明](03%20Multi-Agent%20Zero-Sum%20Soccer%20Game%20Experiment/result_figures/foe_372.png)
![输入图片说明](03%20Multi-Agent%20Zero-Sum%20Soccer%20Game%20Experiment/result_figures/ceq_372.png)

### Experiment 3

In this experiment, I replicated Sutton's Temporal Difference (TD) methods in bounded random walk problems. The results showed that TD(λ) methods generally outperform Widrow-Hoff (TD(1)) in multi-step prediction problems. Additionally, the experiment revealed that the update rule of the weight vector w, the value selection of the learning rate (α), and the value selection of the seed used to generate training data significantly affect the final performance.

![输入图片说明](01%20Implementing%20Methods%20of%20Temporal%20Differences/fig3.png)
![输入图片说明](01%20Implementing%20Methods%20of%20Temporal%20Differences/fig5.png)