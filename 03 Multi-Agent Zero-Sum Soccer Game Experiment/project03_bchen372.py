import numpy as np
import matplotlib.pyplot as plt
import time
from cvxopt import matrix, solvers
from numpy import eye, hstack, ones, vstack, zeros
import copy

class player_A():
    def __init__( self ):
        self.coordinate = np.array([1, 3])
        self.goal = np.array([[1, 1], [2, 1]])
        self.ball = False
        self.first = True
        self.reward = 0

class player_B():
    def __init__( self ):
        self.coordinate = np.array([1, 2])
        self.goal = np.array([[1, 4], [2, 4]])
        self.ball = True
        self.first = False
        self.reward = 0

class soccer_env():
    def __init__( self, player_a, player_b, random_start=True ):
        if random_start:
            new_coordinate = np.array([[1, 2], [1, 3], [2, 2], [2, 3]])
            if np.random.random() <= 0.5:
                a, b = np.random.choice([0, 1], 2, replace=False)# choose 2 sample without replace.
            else:
                a, b = np.random.choice([2, 3], 2, replace=False)
            player_a.coordinate = np.array(new_coordinate[a])
            player_b.coordinate = np.array(new_coordinate[b])
            if np.random.uniform() < 0.5:#previous code is, at the very beginning, each one has 50% probability possessing the ball, now I change to only B get the ball when game starts.
                player_a.ball = False
                player_b.ball = True
            else:
                player_a.ball = False
                player_b.ball = True
        assert (player_a.ball and player_b.ball) == False
        assert (player_a.ball or player_b.ball) == True
        self.done = False
        self.ball = 1 if player_a.ball == True else 0
        self.ball_pos = player_a.coordinate if player_a.ball == True else player_b.coordinate

    def temp_next_step( self, player, action ):
        if action == 0 and player.coordinate[0] == 2:  # meaning North
            temp = copy.deepcopy(player.coordinate)
            temp[0] -= 1
            return np.array(temp)
        elif action == 1 and player.coordinate[0] == 1:  # meaning South
            temp = copy.deepcopy(player.coordinate)
            temp[0] += 1
            return np.array(temp)
        elif action == 2 and player.coordinate[1] > 1:  # meaning West
            temp = copy.deepcopy(player.coordinate)
            temp[1] -= 1
            return np.array(temp)
        elif action == 3 and player.coordinate[1] < 4:  # meaning East
            temp = copy.deepcopy(player.coordinate)
            temp[1] += 1
            return np.array(temp)
        else:# meaning Stick or choosing NEWS but cannot move
            return np.array(player.coordinate)

    def who_first( self, player_a, player_b ):
        if np.random.uniform() <= 0.5:
            player_a.first = True
            player_b.first = False
        else:
            player_a.first = False#you need this step, don't ignore it!
            player_b.first = True

    def first_act( self, player_first, player_second, temp_player_first, temp_player_second ):
        if np.any(temp_player_first != player_second.coordinate):
            player_first.coordinate = temp_player_first
        else:
            player_first.ball = False
            player_second.ball = True
        if np.any(temp_player_second != player_first.coordinate):
            player_second.coordinate = temp_player_second
        else:
            player_first.ball = True
            player_second.ball = False
        return player_first.coordinate, player_second.coordinate, player_first.ball, player_second.ball

    def next_scene( self, player_a, player_b, action_a, action_b ):
        temp_a = self.temp_next_step(player_a, action_a)
        temp_b = self.temp_next_step(player_b, action_b)
        self.who_first(player_a, player_b)
        if player_a.first == True:
            player_a.coordinate, player_b.coordinate, player_a.ball, player_b.ball = self.first_act(player_a, player_b,
                                                                                                    temp_a, temp_b)
        else:
            player_b.coordinate, player_a.coordinate, player_b.ball, player_a.ball = self.first_act(player_b, player_a,
                                                                                                    temp_b, temp_a)
        self.ball = 1 if player_b.ball == True else 0
        self.ball_pos = player_a.coordinate if player_a.ball == True else player_b.coordinate

        if self.ball_pos[1] == player_a.goal[0][1]:#if the second coordinate of ball is == 1, which means the first column, then A wins
            player_a.reward = 100
            player_b.reward = -100
            self.done = True
        elif self.ball_pos[1] == player_b.goal[0][1]:#if the second coordinate of ball is == 4, which means the last column, then B wins
            player_a.reward = -100
            player_b.reward = 100
            self.done = True
        return player_a.reward, player_b.reward, self.done

def plot_error( episode_list, error_list, title="Q Learning", linewidth=1):
    plt.plot(episode_list, error_list, linewidth=linewidth)
    plt.title(title)
    plt.xlim(0, 1e6)
    plt.ylim(0, 0.5)
    plt.yticks(np.arange(0,0.51,0.05))
    plt.xticks(np.arange(0,1e6+1,1e5))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(5,5), useMathText = True)
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-Value Difference')
    plt.show()

def new_game(random_start):
    player_a = player_A()
    player_b = player_B()
    env = soccer_env(player_a, player_b, random_start)
    return player_a, player_b, env


def q_learning():
    episode = 1e6
    epsilon = 0.02
    epsilon_min = 0.01
    epsilon_decay = (epsilon-epsilon_min)/(episode)
    gamma = 0.9

    alpha = 1.
    alpha_end = 0.001

    states = 8
    ball_on_a = 2
    actions = 5

    Q_a = np.zeros((states, states, ball_on_a, actions))#means: state for A, state for B, ball on A, action for A
    Q_b = np.zeros((states, states, ball_on_a, actions))#means: state for A, state for B, ball on A, action for B

    player_a, player_b, env = new_game(random_start=True)

    error_list = []
    episode_list = []

    for i in range(int(episode)):
        assert env.ball == player_a.ball
        pos_a = (player_a.coordinate[0] - 1) * 4 + (player_a.coordinate[1] - 1)
        pos_b = (player_b.coordinate[0] - 1) * 4 + (player_b.coordinate[1] - 1)
        ball = env.ball

        if np.random.random() < epsilon:
            action_a = np.random.choice(actions)
            action_b = np.random.choice(actions)
        else:
            action_a = np.argmax(Q_a[pos_a, pos_b, ball])
            action_b = np.argmax(Q_b[pos_a, pos_b, ball])

        if [pos_a, pos_b, ball, action_a] == [2, 1, 0, 1]:
            prev_q_val = copy.deepcopy(Q_a[2, 1, 0, 1])  # when A on grid#2, B on grid#1, A don't get the ball, A choose South.

        reward_a, reward_b, done = env.next_scene(player_a, player_b, action_a, action_b)
        env.ball = 1 if player_a.ball == True else 0
        new_coordinate_a, new_coordinate_b, new_ball = player_a.coordinate, player_b.coordinate, env.ball
        new_pos_a = (new_coordinate_a[0] - 1) * 4 + (new_coordinate_a[1] - 1)
        new_pos_b = (new_coordinate_b[0] - 1) * 4 + (new_coordinate_b[1] - 1)

        Q_a[pos_a, pos_b, ball, action_a] = (1 - alpha) * Q_a[pos_a, pos_b, ball, action_a] + alpha * ((1 - gamma) * reward_a + gamma * np.max(Q_a[new_pos_a, new_pos_b, new_ball]))\
            if not done else (1 - alpha) * Q_a[pos_a, pos_b, ball, action_a] + alpha * ((1 - gamma) * reward_a + gamma * 0)
        Q_b[pos_a, pos_b, ball, action_b] = (1 - alpha) * Q_b[pos_a, pos_b, ball, action_b] + alpha * ((1 - gamma) * reward_b + gamma * np.max(Q_b[new_pos_a, new_pos_b, new_ball]))\
            if not done else (1 - alpha) * Q_b[pos_a, pos_b, ball, action_b] + alpha * ((1 - gamma) * reward_b + gamma * 0)

         #check:
        assert pos_a != pos_b
        assert new_pos_a != new_pos_b
        assert player_a.ball != player_b.ball
        assert player_a.reward + player_b.reward == 0

        if [pos_a, pos_b, ball, action_a] == [2, 1, 0, 1]: # when A on grid#2, B on grid#1, A don't get the ball, A goes South.
            error_list.append(abs(Q_a[2, 1, 0, 1] - prev_q_val))
            episode_list.append(i)
            print("Episode # ", i, abs(Q_a[2, 1, 0, 1] - prev_q_val))

        if done == True:
            player_a, player_b, env = new_game(random_start=True)

        epsilon -= epsilon_decay
        alpha *= np.e ** (-np.log(alpha/alpha_end)/(1*episode))

    plot_error(episode_list, error_list, title="Q Learner by bchen372", linewidth=0.6)

def friend_q_learning():
    episode = 1e6
    gamma = 0.9

    states = 8
    ball_on_a = 2
    actions = 5

    alpha = 0.3
    alpha_end = 0.001

    Q_a = np.zeros((states, states, ball_on_a, actions, actions))#means: state for A, state for B, ball on A, action for A, action for B

    player_a, player_b, env = new_game(random_start=True)

    error_list = []
    episode_list = []

    for i in range(int(episode)):
        assert env.ball == player_a.ball
        pos_a = (player_a.coordinate[0] - 1) * 4 + (player_a.coordinate[1] - 1)
        pos_b = (player_b.coordinate[0] - 1) * 4 + (player_b.coordinate[1] - 1)
        ball = env.ball

        #on-policy
        action_a = np.random.choice(actions)
        action_b = np.random.choice(actions)

        reward_a, reward_b, done = env.next_scene(player_a, player_b, action_a, action_b)
        env.ball = 1 if player_a.ball == True else 0
        new_coordinate_a, new_coordinate_b, new_ball = player_a.coordinate, player_b.coordinate, env.ball
        new_pos_a = (new_coordinate_a[0] - 1) * 4 + (new_coordinate_a[1] - 1)
        new_pos_b = (new_coordinate_b[0] - 1) * 4 + (new_coordinate_b[1] - 1)

        if not done:
            if player_a.ball == True:
                assert new_pos_a not in [0, 4]
            else:
                assert new_pos_b not in [3, 7]

        if [pos_a, pos_b, ball, action_a, action_b] == [2, 1, 0, 1, 4]:# when A on grid#2, B on grid#1, A don't get the ball, A goes South, B Sticks
            prev_q_val = Q_a[pos_a, pos_b, ball, action_a, action_b]

        Q_a[pos_a, pos_b, ball, action_a, action_b] = (1 - alpha) * Q_a[pos_a, pos_b, ball, action_a, action_b] + alpha * ((1 - gamma) * reward_a + gamma * np.max(Q_a[new_pos_a, new_pos_b, new_ball])) \
            if not done else (1 - alpha) * Q_a[pos_a, pos_b, ball, action_a, action_b] + alpha * ((1 - gamma) * reward_a + gamma * 0)

        #check:
        assert pos_a != pos_b
        assert new_pos_a != new_pos_b
        assert player_a.ball != player_b.ball
        assert player_a.reward + player_b.reward == 0

        if [pos_a, pos_b, ball, action_a, action_b] == [2, 1, 0, 1, 4]:  # ball in B, A go South, B stick?
            error_list.append(abs(Q_a[2, 1, 0, 1, 4] - prev_q_val))
            episode_list.append(i)
            print("Episode # ", i, abs(Q_a[2, 1, 0, 1, 4] - prev_q_val))

        if done == True:
            player_a, player_b, env = new_game(random_start=True)

        alpha *= np.e**(-np.log(alpha/alpha_end)/(1*episode))

    plot_error(episode_list, error_list, title="Friend-Q by bchen372", linewidth=1)

def maximin(q):
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

    M = matrix(q) #the row is the actions available to player A, the column is for player B.
    column_M = M.size[1]
    row_M = M.size[0]

    A = hstack((ones((row_M, 1)), -M)) # v-(x_{1,j} + ... x_{i,j} + x_{5,j}) <= 0, where i is the action from player A, j from player B
    constraint_1 = hstack((zeros((column_M, 1)), -eye(column_M)))  # constraint: p_i > 0 for all i
    A = vstack((A, constraint_1))
    less_than_1 = hstack((0, ones(column_M)))
    larger_than_1 = hstack((0, -ones(column_M)))
    A = matrix(vstack((A, less_than_1, larger_than_1)))  # constraint_2 = vstack(())) # constraint: sum(p_i) == 1

    b = matrix(hstack((zeros(A.size[0] - 2), [1, -1])))
    c = matrix(hstack(([-1], zeros(column_M))))  # objective

    solution = solvers.lp(c, A, b, solver='glpk')

    return solution['x'][0]  # c^T*X, this is equal to -solution['primal objective']

def foe_q_learning():
    np.random.seed(372)

    episode = 1e6
    alpha = 1.
    alpha_end = 0.00005
    gamma = 0.9

    states = 8
    ball_on_a = 2
    actions = 5

    Q_a = np.zeros((states, states, ball_on_a, actions, actions))  # means: state for A, state for B, ball on A, action for A, action for B

    player_a, player_b, env = new_game(random_start=True)

    error_list = []
    episode_list = []

    for i in range(int(episode)):
        pos_a = (player_a.coordinate[0] - 1) * 4 + (player_a.coordinate[1] - 1)
        pos_b = (player_b.coordinate[0] - 1) * 4 + (player_b.coordinate[1] - 1)
        ball = env.ball

        action_a = np.random.choice(actions)
        action_b = np.random.choice(actions)

        if [pos_a, pos_b, ball, action_a, action_b] == [2, 1, 0, 1, 4]:
            prev_q_val = copy.deepcopy(Q_a[2, 1, 0, 1, 4])

        reward_a, reward_b, done = env.next_scene(player_a, player_b, action_a, action_b)
        new_coordinate_a, new_coordinate_b, new_ball = player_a.coordinate, player_b.coordinate, env.ball
        new_pos_a = (new_coordinate_a[0] - 1) * 4 + (new_coordinate_a[1] - 1)
        new_pos_b = (new_coordinate_b[0] - 1) * 4 + (new_coordinate_b[1] - 1)

        equilibrium = maximin(Q_a[pos_a, pos_b, ball])

        #check:
        assert pos_a != pos_b
        assert new_pos_a != new_pos_b
        assert player_a.ball != player_b.ball
        assert player_a.reward + player_b.reward == 0
        if not done:
            if player_a.ball == True:
                assert new_pos_a not in [0, 4]
            else:
                assert new_pos_b not in [3, 7]
        if done:
            if player_a.ball == True and player_a.reward == 100:
                assert new_pos_a in [0, 4]
            elif player_a.ball == True and player_a.reward == -100:
                assert new_pos_a in [3, 7]
            elif player_b.ball == True and player_b.reward == 100:
                assert new_pos_b in [3, 7]
            elif player_b.ball == True and player_b.reward == -100:
                assert new_pos_b in [0, 4]

        Q_a[pos_a, pos_b, ball, action_a, action_b] = (1 - alpha) * Q_a[pos_a, pos_b, ball, action_a, action_b] + alpha * ((1 - gamma) * reward_a + gamma * equilibrium)

        if [pos_a, pos_b, ball, action_a, action_b] == [2, 1, 0, 1, 4]:  # ball in B, A go South, B stick?
            error_list.append(abs(Q_a[2, 1, 0, 1, 4] - prev_q_val))
            episode_list.append(i)
            print("Episode # ", i, abs(Q_a[2, 1, 0, 1, 4] - prev_q_val))

        alpha *= np.e ** (-np.log(alpha/alpha_end)/(1*episode))

        if env.done == True:
            player_a, player_b, env = new_game(random_start=True)

    print(Q_a[2, 1, 0])
    plot_error(episode_list, error_list, title="Foe-Q by bchen372", linewidth=1)

def CE( q_a, q_b ):
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    M = matrix(q_a)
    column_M = M.size[1]

    A = np.zeros((column_M * (column_M - 1), (column_M * column_M)))
    B = np.zeros((column_M * (column_M - 1), (column_M * column_M)))

    q_a = np.array(q_a)
    q_b = np.array(q_b)

    # system of equations for correlated equilibrium q-learning
    row = 0
    for i in range(column_M):
        for j in range(column_M):
            if i != j:
                # given a fixed action m from player B, we need (u_{i,m} - u_{j,m})* x_{i,m} >= 0, for any i,j
                # where i,j are actions available to player A, x is the correlated equilibrium we want to maximize.
                A[row, i * column_M:(i + 1) * column_M] = q_a[i] - q_a[j]
                row += 1
    row = 0
    for i in range(column_M):
        for j in range(column_M):
            if i != j:
                # given a fixed action n from player A, we need (u_{n,i} - u_{n,j})* x_{n,i} >= 0, for any i,j
                # where i,j now are actions available to player B, x is the correlated equilibrium we want to maximize.
                B[row, i:(column_M * column_M):column_M] = q_b[:, i] - q_b[:, j]
                row += 1
    A = matrix(vstack((A, B)))

    A = hstack((ones((A.size[0], 1)), A))
    constraint_1 = hstack((zeros((column_M ** 2, 1)), -eye(column_M ** 2)))  # constraint: p_i > 0 for all i
    A = vstack((A, constraint_1))
    constraint_2 = vstack(
        (hstack((0, ones(column_M ** 2))), hstack((0, -ones(column_M ** 2)))))  # constraint: sum(p_i) == 1
    A = matrix(vstack((A, constraint_2)))

    b = matrix(hstack((zeros(A.size[0] - 2), [1, -1])))

    c = matrix(hstack(([-1.], -(q_a + q_b).flatten())))  # objective, CE maximizes the sum of both agents' rewards

    solution = solvers.lp(c, A, b, solver='glpk')
    if solution['x'] is None: return 0, 0 #sometimes solution is None.
    # calculate correlated equilibrium
    qa_expected = np.matmul(q_a.flatten(), solution['x'][1:])[0]
    qb_expected = np.matmul(q_b.transpose().flatten(), solution['x'][1:])[0]

    return qa_expected, qb_expected


def CE_q_learning():
    np.random.seed(372)
    episode = 1e6
    alpha = 1.
    alpha_end = 0.00005
    gamma = 0.9

    states = 8
    ball_on_a = 2
    actions = 5

    Q_a = np.zeros((states, states, ball_on_a, actions, actions))  # means: state for A, state for B, ball on A, action for A, action for B
    Q_b = np.zeros((states, states, ball_on_a, actions, actions))  # means: state for A, state for B, ball on A, action for A, action for B

    player_a, player_b, env = new_game(random_start=True)

    error_list = []
    episode_list = []

    for i in range(int(episode)):
        pos_a = (player_a.coordinate[0] - 1) * 4 + (player_a.coordinate[1] - 1)
        pos_b = (player_b.coordinate[0] - 1) * 4 + (player_b.coordinate[1] - 1)
        ball = env.ball

        action_a = np.random.choice(actions)
        action_b = np.random.choice(actions)

        if [pos_a, pos_b, ball, action_a, action_b] == [2, 1, 0, 1, 4]:# when A on grid#2, B on grid#1, A don't get the ball, A goes South, B Sticks
            prev_q_val = Q_a[2, 1, 0, 1, 4]

        reward_a, reward_b, done = env.next_scene(player_a, player_b, action_a, action_b)
        new_coordinate_a, new_coordinate_b, new_ball = player_a.coordinate, player_b.coordinate, env.ball
        new_pos_a = (new_coordinate_a[0] - 1) * 4 + (new_coordinate_a[1] - 1)
        new_pos_b = (new_coordinate_b[0] - 1) * 4 + (new_coordinate_b[1] - 1)
        objective_a, objective_b = CE(Q_a[pos_a, pos_b, ball], Q_b[pos_a, pos_b, ball])

        # check:
        assert pos_a != pos_b
        assert new_pos_a != new_pos_b
        assert player_a.ball != player_b.ball
        assert player_a.reward + player_b.reward == 0
        if not done:
            if player_a.ball == True:
                assert new_pos_a not in [0, 4]
            else:
                assert new_pos_b not in [3, 7]
        if done:
            if player_a.ball == True and player_a.reward == 100:
                assert new_pos_a in [0, 4]
            elif player_a.ball == True and player_a.reward == -100:
                assert new_pos_a in [3, 7]
            elif player_b.ball == True and player_b.reward == 100:
                assert new_pos_b in [3, 7]
            elif player_b.ball == True and player_b.reward == -100:
                assert new_pos_b in [0, 4]

        Q_a[pos_a, pos_b, ball, action_a, action_b] = (1 - alpha) * Q_a[pos_a, pos_b, ball, action_a, action_b] + alpha * ((1 - gamma) * reward_a + gamma * objective_a)
        Q_b[pos_a, pos_b, ball, action_a, action_b] = (1 - alpha) * Q_b[pos_a, pos_b, ball, action_a, action_b] + alpha * ((1 - gamma) * reward_b + gamma * objective_b)

        if [pos_a, pos_b, ball, action_a, action_b] == [2, 1, 0, 1, 4]:  # ball in B, A go South, B stick?
            error_list.append(abs(Q_a[2, 1, 0, 1, 4] - prev_q_val))
            episode_list.append(i)
            print("Episode # ", i, abs(Q_a[2, 1, 0, 1, 4] - prev_q_val))

        alpha *= np.e ** (-np.log(alpha/alpha_end)/(1*episode))

        if done == True:
            player_a, player_b, env = new_game(random_start=True)
    print(Q_a[2, 1, 0])
    plot_error(episode_list, error_list, title="Correlated-Q by bchen372", linewidth=1)


if __name__ == "__main__":
    start_time = time.time()
    #q_learning()
    #friend_q_learning()
    #foe_q_learning()
    #CE_q_learning()
    print('Time used:', time.time()-start_time)


