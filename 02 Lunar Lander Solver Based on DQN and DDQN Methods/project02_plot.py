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

def analysis_alpha(alpha_path_list, alpha_list):
    for i, path in enumerate(alpha_path_list):
        with open(path) as f:
            lines = f.readlines()
            content = [line.split() for line in lines]
            start_index = 8
            end_index = len(content) - 5
            value = content[start_index:end_index]
            episode = []
            average_reward = []
            for item in value:
                episode.append(float(item[0]))
                average_reward.append(float(item[3]))
            plt.plot(episode, average_reward, linewidth='1', label='$\\alpha$ = %s' % alpha_list[i])
    plt.xlabel('Episode')
    plt.ylabel('Average Reward of Last 100 Episodes')
    plt.title('Analysis of Hyper-parameter $\\alpha$ when $\gamma$ = 0.99' )
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def analysis_gamma(gamma_path_list, alpha_list):
    for i, path in enumerate(gamma_path_list):
        with open(path) as f:
            lines = f.readlines()
            content = [line.split() for line in lines]
            start_index = 8
            end_index = len(content) - 5
            value = content[start_index:end_index]
            episode = []
            average_reward = []
            for item in value:
                episode.append(float(item[0]))
                average_reward.append(float(item[3]))
            plt.plot(episode, average_reward, linewidth='0.6', label='$\\alpha$ = %s' % alpha_list[i])
    plt.xlabel('Episode')
    plt.ylabel('Average Reward of Last 100 Episodes')
    plt.title('Analysis of Hyper-parameter $\\alpha$ when $\gamma$ = 0.59')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def analysis_epsilon_decay(decay_path_list, decay_list):
    for i, path in enumerate(decay_path_list):
        with open(path) as f:
            lines = f.readlines()
            content = [line.split() for line in lines]
            start_index = 8
            end_index = len(content) - 5
            value = content[start_index:end_index]
            episode = []
            average_reward = []
            for item in value:
                episode.append(float(item[0]))
                average_reward.append(float(item[3]))
            plt.plot(episode, average_reward, linewidth='1', label='Epsilon Decay = %s' % decay_list[i])
    plt.xlabel('Episode')
    plt.ylabel('Average Reward of Last 100 Episodes')
    plt.title('Analysis of Hyper-parameter Epsilon Decay')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def analysis_sample_size(sample_path_list, sample_list):
    for i, path in enumerate(sample_path_list):
        with open(path) as f:
            lines = f.readlines()
            content = [line.split() for line in lines]
            start_index = 8
            end_index = len(content) - 5
            value = content[start_index:end_index]
            episode = []
            average_reward = []
            for item in value:
                episode.append(float(item[0]))
                average_reward.append(float(item[3]))
            plt.plot(episode, average_reward, linewidth='0.9', label='Sample Size = %s' % sample_list[i])
    plt.xlabel('Episode')
    plt.ylabel('Average Reward of Last 100 Episodes')
    plt.title('Analysis of Hyper-parameter Sample Size')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def analysis_update_step(update_path_list, update_list):
    for i, path in enumerate(update_path_list):
        with open(path) as f:
            lines = f.readlines()
            content = [line.split() for line in lines]
            start_index = 8
            end_index = len(content) - 5
            value = content[start_index:end_index]
            episode = []
            average_reward = []
            for item in value:
                episode.append(float(item[0]))
                average_reward.append(float(item[3]))
            plt.plot(episode, average_reward, linewidth='1.5', label='Update Step = %s' % update_list[i])
    plt.xlabel('Episode')
    plt.ylabel('Average Reward of Last 100 Episodes')
    plt.title('Analysis of Hyper-parameter Update Step')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_alpha():
    ###plot figure 3. Analysis of Hyper-parameter: alpha
    ###alpha = [0.0005, 0.001, 0.0015, 0.002, 0.0025], gamma = 0.99, sample size = 30, epsilon decay = 150, update step = 1
    alpha_01 = './test alpha/Agent #1 Train-19-03-16 14-22-08.txt'
    alpha_02 = './test alpha/Agent #2 Train-19-03-16 14-47-43.txt'
    alpha_03 = './test alpha/Agent #3 Train-19-03-16 15-00-29.txt'
    alpha_04 = './test alpha/Agent #4 Train-19-03-16 15-18-33.txt'
    alpha_05 = './test alpha/Agent #5 Train-19-03-16 15-36-33.txt'
    alpha_path_list = [alpha_01, alpha_02, alpha_03, alpha_04, alpha_05]
    alpha_list = [0.0005 * i for i in range(1, 6)]
    analysis_alpha(alpha_path_list, alpha_list)

def plot_gamma():
    ###plot figure 3. Analysis of Hyper-parameter: gamma
    ### alpha = [0.0005 * i for i in range(1, 9)], gamma = 0.59, sample size = 30, epsilon decay = 150, update step = 1
    gamma_01 = './test gamma/Agent #1 Train-19-03-16 02-00-38.txt'
    gamma_02 = './test gamma/Agent #2 Train-19-03-16 03-26-11.txt'
    gamma_03 = './test gamma/Agent #3 Train-19-03-16 04-49-52.txt'
    gamma_04 = './test gamma/Agent #4 Train-19-03-16 06-13-18.txt'
    gamma_05 = './test gamma/Agent #5 Train-19-03-16 07-32-35.txt'
    gamma_06 = './test gamma/Agent #6 Train-19-03-16 08-53-50.txt'
    gamma_07 = './test gamma/Agent #7 Train-19-03-16 10-17-39.txt'
    gamma_08 = './test gamma/Agent #8 Train-19-03-16 11-47-53.txt'
    gamma_path_list = [gamma_01, gamma_02, gamma_03, gamma_04, gamma_05, gamma_06, gamma_07, gamma_08]
    alpha_list = [0.0005 * i for i in range(1, 9)]
    analysis_gamma(gamma_path_list, alpha_list)

def plot_epsilon_decay():
    ###plot figure 5. Analysis of Hyper-parameter: epsilon decay
    ### alpha = 0.001, gamma = 0.59, sample size = 30, epsilon_decay = [100 + 50 * i for i in range(1, 7)], update step = 1
    d_01 = './test epsilon decay/Agent #1 Train-19-03-16 18-16-18.txt'
    d_02 = './test epsilon decay/Agent #2 Train-19-03-16 18-38-23.txt'
    d_03 = './test epsilon decay/Agent #3 Train-19-03-16 19-27-56.txt'
    d_04 = './test epsilon decay/Agent #4 Train-19-03-16 19-44-01.txt'
    d_05 = './test epsilon decay/Agent #5 Train-19-03-16 20-23-41.txt'
    d_06 = './test epsilon decay/Agent #6 Train-19-03-16 20-52-07.txt'
    d_07 = './test epsilon decay/Agent #7 Train-19-03-16 21-08-54.txt'
    decay_path_list = [d_01, d_02, d_03, d_04, d_05, d_06, d_07]
    decay_list = [100 + 50 * i for i in range(1, 8)]
    analysis_epsilon_decay(decay_path_list, decay_list)

def plot_sample_size():
    ### plot figure 6. Analysis of Hyper-parameter: sample size
    ### alpha = 0.001, gamma = 0.59, sample size = [20 + 10 * i for i in range(1, 7)], epsilon_decay = 250, update step = 1
    d_01 = './test sample size/Agent #1 Train-19-03-16 22-05-05.txt'
    d_02 = './test sample size/Agent #2 Train-19-03-16 22-40-13.txt'
    d_03 = './test sample size/Agent #3 Train-19-03-16 22-52-29.txt'
    d_04 = './test sample size/Agent #4 Train-19-03-16 23-06-34.txt'
    d_05 = './test sample size/Agent #5 Train-19-03-17 05-25-38.txt'
    d_06 = './test sample size/Agent #6 Train-19-03-17 08-43-08.txt'
    sample_path_list = [d_01, d_02, d_03, d_04, d_05, d_06]
    sample_list = [20 + 10 * i for i in range(1, 7)]
    analysis_sample_size(sample_path_list, sample_list)

def plot_update_step():
    ### plot figure 6. Analysis of Hyper-parameter: sample size
    ### alpha = 0.001, gamma = 0.59, sample size = [20 + 10 * i for i in range(1, 7)], epsilon_decay = 250, update step = 1
    d_01 = './test update step/Agent #1 Train-19-03-17 10-24-32.txt'
    d_02 = './test update step/Agent #2 Train-19-03-17 10-45-12.txt'
    d_03 = './test update step/Agent #3 Train-19-03-17 10-57-44.txt'
    d_04 = './test update step/Agent #4 Train-19-03-17 11-08-52.txt'
    update_path_list = [d_01, d_02, d_03, d_04]
    update_list = [1, 2, 5, 10]
    analysis_update_step(update_path_list, update_list)


def plot_figure():
    path_dqn = './plot/Agent #1 Train-19-03-16 00-43-31.txt'
    path_ddqn = './plot/Agent #1 Train-19-03-17 18-52-36.txt'
    path_list = [path_dqn, path_ddqn]
    for path in path_list:
        with open(path) as f:
            lines = f.readlines()
            content = [line.split() for line in lines]
            start_index = 8
            end_index = len(content) - 5
            value = content[start_index:end_index]
            episode = []
            episode_reward = []
            average_reward = []
            for item in value:
                episode.append(float(item[0]))
                episode_reward.append(float(item[2]))
                average_reward.append(float(item[3]))
            mean = [np.mean(episode_reward)] * len(episode)
            p1, = plt.plot(episode, episode_reward, linewidth = '0.5')
            p2, = plt.plot(episode, average_reward, linewidth = '1.5')
            p3, = plt.plot(episode, mean, linewidth = '1.5')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('DDQN Method: Reward per episode and reward per trial for 100 trials')
            plt.grid(True)
            plt.legend([p1, p2, p3], ['Reward Per Episode','Average Reward Per 100 Episodes', 'Average Reward for All Episodes'])
            plt.show()

def plot_test():
    #dqn method, alpha = 0.0010, epsilon decay = 150, sample size = 30, update step = 1
    path_alpha_01 = './test result/Agent #0 Test-19-03-17 18-29-40.txt'
    #dqn method, alpha = 0.0015, epsilon decay = 150, sample size = 30, update step = 1
    path_alpha_015 = './test result/Agent #0 Test-19-03-17 18-48-07.txt'
    #dqn method, alpha = 0.0010, epsilon decay = 200, sample size = 30, update step = 1
    path_decay200 = './test result/Agent #0 Test-19-03-17 19-24-51.txt'
    #dqn method, alpha = 0.0010, epsilon decay = 150, sample size = 50, update step = 1
    path_sample50 = './test result/Agent #0 Test-19-03-17 23-41-47.txt'
    #dqn method, alpha = 0.0010, epsilon decay = 150, sample size = 30, update step = 5
    path_update5 = './test result/Agent #0 Test-19-03-18 00-09-57.txt'
    # ddqn methods
    path_ddqn = './test result/Agent #0 Test-19-03-17 23-30-09.txt'
    path_list = [path_alpha_01, path_alpha_015, path_decay200, path_sample50, path_update5, path_ddqn]
    for path in path_list:
        with open(path) as f:
            lines = f.readlines()
            content = [line.split() for line in lines]
            start_index = 8
            end_index = len(content) - 5
            value = content[start_index:end_index]
            episode = []
            episode_reward = []
            average_reward = []
            for item in value:
                episode.append(float(item[0]))
                episode_reward.append(float(item[2]))
                average_reward.append(float(item[3]))
            mean = [np.mean(episode_reward)] * len(episode)
            p1, = plt.plot(episode, episode_reward, linewidth = '0.7', color = 'y')
            p2, = plt.plot(episode, average_reward, linewidth = '1.5', color = 'r')
            p3, = plt.plot(episode, mean, linewidth = '1.5', color = 'g')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Test')
            plt.grid(True)
            plt.legend([p1, p2, p3], ['Reward Per Test','Average Reward Per 100 Tests', 'Average Reward for All Tests'])
            plt.show()


if __name__ == "__main__":
    #####################################################
    ###plot figure 2. Reward at each episode and Reward per trial for 100 trial
    plot_figure()
    #####################################################

    #####################################################
    ###plot figure 3. Analysis of Hyper-parameter: alpha
    #plot_alpha()
    #####################################################


    #####################################################
    ### plot figure 4. Analysis of Hyper-parameter: gamma
    #plot_gamma()
    #####################################################


    #####################################################
    ### plot figure 5. Analysis of Hyper-parameter: epsilon decay
    #plot_epsilon_decay()
    #####################################################


    #####################################################
    ### plot figure 6. Analysis of Hyper-parameter: sample size
    #plot_sample_size()
    #####################################################


    #####################################################
    ### plot figure 7. Analysis of Hyper-parameter: update step
    #plot_update_step()
    #####################################################

    #####################################################
    ### plot figure 8-9. Tests
    #plot_test()
    #####################################################


