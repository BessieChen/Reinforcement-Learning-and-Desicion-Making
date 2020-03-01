import gym, random, tempfile
import numpy as np
from gym import wrappers
from collections import deque
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD
import os
from datetime import datetime
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Agent():
    Agent_th = 0
    def __init__(self, feature_of_states, number_of_actions, alpha, gamma, epsilon,
                 max_buffer_length, training_size, step_update_target, test, weight_loading_path):
        if not test:
            Agent.Agent_th += 1
        self.alpha = alpha
        self.gamma = gamma
        self.feature_of_states = feature_of_states
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.buffer = deque(maxlen=max_buffer_length)
        self.training_size = training_size
        self.step_update_target = step_update_target
        self.result_path = '/Users/bessie.chan/Downloads/Reinforcement/homework/result_from_Beixi/'
        self.train_file_path = self.result_path + 'Train file/'
        self.test_file_path = self.result_path + 'Test file/'
        self.weight_file_path = self.result_path + 'Weight file/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.train_file_path):
            os.makedirs(self.train_file_path)
        if not os.path.exists(self.test_file_path):
            os.makedirs(self.test_file_path)
        if not os.path.exists(self.weight_file_path):
            os.makedirs(self.weight_file_path)

        if test:
            if not weight_loading_path:
                raise ValueError('Weight path is not given')
            if not os.path.exists(weight_loading_path):
                raise ValueError('Weight path must exist')
            self.model.load_weights(weight_loading_path)
        self.train_path = self.train_file_path + 'Agent #' + str(Agent.Agent_th) + ' Train' + datetime.now().strftime("-%y-%m-%d %H-%M-%S")  + '.txt'
        self.test_path = self.test_file_path + 'Agent #' + str(Agent.Agent_th) + ' Test' + datetime.now().strftime("-%y-%m-%d %H-%M-%S")  + '.txt'
        self.weight_path = self.weight_file_path + 'Agent #' + str(Agent.Agent_th) + ' Weight' + datetime.now().strftime("-%y-%m-%d %H-%M-%S") + '.h5'

    def choose_action( self, state ):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.number_of_actions)
        return np.argmax(self.model.predict(state)[0])

    def build_model(self):
        temp_model = Sequential()
        temp_model.add(Dense(60, activation="relu", input_dim=self.feature_of_states))
        temp_model.add(Dense(60, activation="relu"))
        temp_model.add(Dense(self.number_of_actions, activation="linear"))
        temp_model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return temp_model

    def add_to_buffer( self, state, action, reward, next_state, done ):
        self.buffer.append((state, action, reward, next_state, done))

    def replay_dqn( self ):
        sample = random.sample(self.buffer, self.training_size)
        sample = np.array(sample)
        curent_state_vector = sample[:, 0]
        actions = np.array(sample[:, 1], dtype=int)
        reward_vector = sample[:, 2]
        next_state_vector = sample[:, 3]
        done = sample[:, 4]
        y_target = self.model.predict_on_batch(np.vstack(curent_state_vector))
        q_from_target = self.target_model.predict_on_batch(np.vstack(next_state_vector))
        y_target[range(self.training_size), actions] = reward_vector + np.logical_not(done) * np.multiply(
            self.gamma, np.max(q_from_target, axis=1))

        return self.model.train_on_batch(np.vstack(curent_state_vector), y_target)

    def replay_ddqn(self):
        sample = random.sample(self.buffer, self.training_size)
        sample = np.array(sample)
        curent_state_vector = sample[:, 0]
        actions = np.array(sample[:, 1], dtype=int)
        reward_vector = sample[:, 2]
        next_state_vector = sample[:, 3]
        done = sample[:, 4]
        action_from_current_model = np.argmax(self.model.predict_on_batch(np.vstack(next_state_vector)), axis=1)
        q_from_target = self.target_model.predict_on_batch(np.vstack(next_state_vector))[range(self.training_size),action_from_current_model][0]
        y_target = self.model.predict_on_batch(np.vstack(curent_state_vector))
        y_target[range(self.training_size), actions] = reward_vector + np.logical_not(done)*np.multiply(self.gamma, q_from_target)

        return self.model.train_on_batch(np.vstack(curent_state_vector), y_target)



def first_train( max_buffer_length = 2000000, training_size = 50, alpha = 0.0015, gamma = 0.99, epsilon_start = 0.9999
    , epsilon_end =0.001, epsilon_decay = 150, max_episode = 2000, max_step = 1000, step_update_target = 2 ,seed = None,
    ddqn = False, test = False, weight_loading_path = ''):
    env = gym.make('LunarLander-v2')
    if ddqn == True: #ddqn methods need to explore more.
        alpha = 0.0001
        epsilon_start = 1
        epsilon_end = 0.01
        epsilon_decay = 6000
        max_episode = 20000
    if seed:
        env.seed(seed)

    if test:
        tdir = tempfile.mkdtemp()
        env.render()
        env = wrappers.Monitor(env, tdir, force=True, video_callable=None)

    feature_of_states = env.observation_space.shape[0]
    number_of_actions = env.action_space.n
    epsilon = epsilon_start
    total_reward = deque(maxlen = 100)
    np.set_printoptions(precision=2)
    agent = Agent(feature_of_states, number_of_actions, alpha, gamma, epsilon, max_buffer_length,
                  training_size, step_update_target, test, weight_loading_path)
    start_time = datetime.now()

    if not test: #if test == True, it means you already get the .h5 file saving weights
        file = open(agent.train_path, 'w+')
    else:
        file = open(agent.test_path, 'w+')

    file.write("{:^10}\n{:%Y-%m-%d %H:%M}\n\n".format('Start Time:', start_time))
    file.write('{}{}\n'.format('Agent #',Agent.Agent_th))
    file.write('{:^10}{:^10}{:^10}{:^10}{:^8}\n'.format('Alpha','Gamma','Epsilon','Sample','Update'))
    file.write("{:^10.4f}{:^10.3f}{:^10.3f}{:^10}{:^8}\n\n".format(agent.alpha, agent.gamma, epsilon_decay, agent.training_size, agent.step_update_target))
    file.write("{:^8}{:^10}{:^16}{:^16}{:^16}{:^16}\n".format('Episodes', 'Epsilon', 'Episode_Reward', 'Average_Reward', 'Training_Loss',
                                                        'Episode_Step'))
    if test:
        max_episode = 500
    for episode in range(1,max_episode+1):
        episode_reward = 0
        score = 0
        state = env.reset().reshape([1, feature_of_states])
        for count_step in range(max_step):

            if not test:
                action = agent.choose_action(state)
            else: action = np.argmax(agent.model.predict(state)[0])

            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape([1, feature_of_states])
            agent.add_to_buffer(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if not test:
                if len(agent.buffer) >= agent.training_size:
                    if not ddqn:
                        score += agent.replay_dqn()
                    else: score += agent.replay_ddqn()
                if count_step % agent.step_update_target == 0:
                    agent.target_model.set_weights(agent.model.get_weights())

            if done:
                total_reward.append(episode_reward)
                avg_reward = np.mean(total_reward)
                file.write("{:^8}{:^10.3f}{:^16.3f}{:^16.3f}{:^16.3f}{:^16}\n".format(episode, agent.epsilon, episode_reward, avg_reward, score, count_step))
                if episode % 10 == 0:
                    print('Episode', episode, 'Alpha',agent.alpha, 'Gamma', agent.gamma, 'Epsilon',
                          '%.3f' %agent.epsilon, 'Sample', agent.training_size,
                          'Update', agent.step_update_target,  'Reward','%.2f' %episode_reward,
                          'Average Reward','%.2f' %avg_reward)
                break

        if not test:
            if avg_reward > 200:
                print('Completed training at episode: ', episode)
                agent.model.save_weights(agent.weight_path)
                print("Saved weights to computer")
                break
            if agent.epsilon > epsilon_end:
                agent.epsilon -= (epsilon_start - epsilon_end) / epsilon_decay
    file.write("\n{:^10}\n{:%Y-%m-%d %H:%M}\n{:^10}\n{}".format('End Time:', datetime.now(), 'Time Used:',datetime.now()-start_time))
    file.close()

def train_default_function():
    start_time = datetime.now()
    print('start_time', start_time)
    first_train()
    end_time = datetime.now()
    print("end_time", end_time)


def hyper_parameter_analysis():
    start_time = datetime.now()
    print('start_time',start_time)
    alpha_list = [0.0005 * i for i in range(1, 7)]
    epsilon_decay_list = [100 + 50 * i for i in range(1, 7)]
    training_size_list = [20 + 10 * i for i in range(1, 7)]
    step_update_list = [1, 2, 5, 10]
    gamma_list = [0.99, 0.79]

    for gamma in gamma_list:
        for step_update_target in step_update_list:
            for training_size in training_size_list:
                for epsilon_decay in epsilon_decay_list:
                    for alpha in alpha_list:
                        start_time_2 = datetime.now()
                        first_train(training_size=training_size, alpha=alpha, gamma=gamma,
                                    epsilon_decay=epsilon_decay, step_update_target=step_update_target, ddqn=True)
                        print('start_time', start_time_2, 'End Time:', datetime.now(), 'Time Used:',
                              datetime.now() - start_time_2)

    end_time = datetime.now()
    print("end_time", end_time)

def test_h5_file():
    # dqn method, alpha = 0.0010, epsilon decay = 150, sample size = 30, update step = 1
    path_alpha_01 = './test h5 file/Agent #2 Weight-19-03-16 14-47-43.h5'
    # dqn method, alpha = 0.0015, epsilon decay = 150, sample size = 30, update step = 1
    path_alpha_015 = './test h5 file/Agent #3 Weight-19-03-16 15-00-29.h5'
    # dqn method, alpha = 0.0010, epsilon decay = 200, sample size = 30, update step = 1
    path_decay200 = './test h5 file/Agent #2 Weight-19-03-16 18-38-23.h5'
    # dqn method, alpha = 0.0010, epsilon decay = 150, sample size = 50, update step = 1
    path_sample50 = './test h5 file/Agent #1 Weight-19-03-17 22-30-07.h5'
    # dqn method, alpha = 0.0010, epsilon decay = 150, sample size = 30, update step = 5
    path_update5 = './test h5 file/Agent #2 Weight-19-03-17 22-41-52.h5'
    # ddqn method
    path_ddqn = './test h5 file/Agent #1 Weight-19-03-17 21-16-06.h5'
    for path in [path_alpha_01, path_alpha_015, path_decay200, path_sample50, path_update5, path_ddqn]:
        first_train(test=True, weight_loading_path=path)


if __name__ == "__main__":

    ###if you want to train the default training model:
    train_default_function()

    ###if you want to pass your own hyper parameters to train agent:
    ##parameters you can pass in:
    #1.max_buffer_length = 2000000, 2.training_size = 50, 3.alpha = 0.0015, 4.gamma = 0.99, 5.epsilon_start = 0.9999
    #6.epsilon_end =0.001, 7.epsilon_decay = 150, 8.max_episode = 2000, 9.max_step = 1000, 10.step_update_target = 2
    #11.seed = None -- seed for environment
    #12.ddqn = False -- if you want to use ddqn method or not, note that it would take a very long time!
    #13.test = False -- remain false if you just want to train your model
    #first_train(alpha = 0.0001, training_size = 30, step_update_target = 1, ddqn = True)

    ###if you want to train whole set of hyper-parameters: (not recommend! it needs a huge amount of time!)
    #hyper_parameter_analysis()

    ###if you want to test my saved model or your own saved model:
    #first_train(test=True, weight_loading_path='./test h5 file/Agent #1 Weight-19-03-17 21-16-06.h5')
    
