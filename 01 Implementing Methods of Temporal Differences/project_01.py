import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm,trange

se_ed = 11
np.random.seed(se_ed)

def generate_data(training_size, sequence_size):
    whole_training_set = []
    for _ in range(training_size):
        one_training_set = []
        for _ in range(sequence_size):
            state = [3]
            if_end = False
            while not if_end:
                state.append(state[-1] + np.random.choice([-1, 1]))
                if (state[-1] == 0) or (state[-1] == 6):
                    if_end = True
            x_vector = []
            for k in range(len(state) - 1):
                x = [0., 0., 0., 0., 0.]
                x[state[k] - 1] = 1.
                x_vector.append(x)
            one_training_set.append(x_vector)
        whole_training_set.append(one_training_set)
    return whole_training_set


def compute(whole_training_set, training_size, sequence_size, alpha, lambda_, epsilon):
    truth = [1.0/6, 2.0/6, 3.0/6, 4.0/6, 5.0/6]
    error_list = []
    weight_list = []
    for training_n in range(training_size):
        weight = [0.5, 0.5, 0.5, 0.5, 0.5]
        converge = False
        train = whole_training_set[training_n]
        while not converge:
            dw = 0.
            for m in range(sequence_size):
                x_vector = train[m]
                P_last = 1 if x_vector[-1][-1] == 1.0 else 0
                for t in range(1,len(x_vector)+1):
                    P_current = np.dot(weight, x_vector[t-1])
                    P_next = P_last if t == (len(x_vector)) else np.dot(weight, x_vector[t])
                    derivative = 0.
                    for k in range(1,t+1):
                        derivative = np.add(np.multiply(np.power(lambda_, t - k), x_vector[k-1]), derivative)
                    temp = np.multiply(alpha*(P_next - P_current), derivative)
                    dw += temp
            converge = np.all(np.abs(dw) < epsilon)
            weight += dw
        weight_list.append(weight)
        error = np.sqrt(np.mean(np.power(np.subtract(truth, weight), 2)))
        error_list.append(error)
    mean_error = np.mean(error_list)
    return mean_error


def compute_without_converge(whole_training_set, training_size, sequence_size, alpha, lambda_):
    truth = [1.0/6, 2.0/6, 3.0/6, 4.0/6, 5.0/6]
    error_list = []
    for training_n in range(training_size):
        weight = [0.5, 0.5, 0.5, 0.5, 0.5]
        train = whole_training_set[training_n]
        for m in range(sequence_size):
            x_vector = train[m]
            dw = 0.
            P_last = 1. if x_vector[-1][-1] == 1.0 else 0.
            for t in range(1,len(x_vector)+1):
                P_current = np.dot(weight, x_vector[t-1])
                P_next = P_last if t == (len(x_vector)) else np.dot(weight, x_vector[t])
                derivative = 0.
                for k in range(1, t+1):
                    derivative = np.add(np.multiply(np.power(lambda_, t - k), x_vector[k-1]), derivative)
                temp = np.multiply(alpha*(P_next - P_current), derivative)
                dw += temp
            weight += dw
            error = np.sqrt(np.average(np.power(np.subtract(truth, weight), 2)))
            error_list.append(error)
    return np.mean(error_list)


def figure_3():
    lambda_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    alpha = 0.01
    epsilon = 0.001
    training_size = 100
    sequence_size = 10

    rsme = []
    whole_training_set = generate_data(training_size, sequence_size)
    for i in trange(len(lambda_list)):
        tqdm.write("Done task %i" % i)
        error = compute(whole_training_set, training_size, sequence_size, alpha, lambda_list[i], epsilon)
        rsme.append(error)

    plt.figure(num=None, figsize=(6, 6), dpi=80)
    plt.plot(lambda_list, rsme, marker='o')
    plt.xlabel('$\lambda$')
    plt.ylabel('ERROR')
    plt.title('Figure 3')
    plt.text(.9, .165, "Widrow-Hoff", ha="center", va="center", rotation=0, size=15)
    plt.grid(True)
    plt.show()


def figure_4():
    lambda_list = [0.0,0.3,0.8,1.0]
    alpha_list = np.linspace(0., 0.6, 13)
    training_size = 100
    sequence_size = 10

    rsme = []
    whole_training_set = generate_data(training_size, sequence_size)
    for i in trange(len(lambda_list)):
        tqdm.write("Done task %i" % i)
        rsme_alpha = []
        for alpha in alpha_list:
            error = compute_without_converge(whole_training_set, training_size, sequence_size, alpha, lambda_list[i])
            rsme_alpha.append(error)
        rsme.append(rsme_alpha)

    p1, = plt.plot(alpha_list, rsme[0], marker='o')
    p2, = plt.plot(alpha_list, rsme[1], marker='o')
    p3, = plt.plot(alpha_list, rsme[2], marker='o')
    p4, = plt.plot(alpha_list, rsme[3], marker='o')
    plt.xlabel('$\\alpha$')
    plt.ylabel('ERROR')
    plt.title('Figure 4')
    plt.grid(True)
    plt.legend([p1, p2, p3, p4], ['$\lambda$ = 0.0', '$\lambda$ = 0.3', '$\lambda$ = 0.8', '$\lambda$ = 1.0'])
    plt.show()



def figure_5():
    lambda_list = np.linspace(0,1,11)
    alpha_list = np.linspace(0., 0.6, 13)
    training_size = 100
    sequence_size = 10

    rsme = []
    whole_training_set = generate_data(training_size, sequence_size)
    for i in trange(len(lambda_list)):
        tqdm.write("Done task %i" % i)
        rsme_alpha = []
        for alpha in alpha_list:
            error = compute_without_converge(whole_training_set, training_size, sequence_size, alpha, lambda_list[i])
            rsme_alpha.append(error)
        min_error = min(rsme_alpha)
        rsme.append(min_error)

    plt.figure(num=None, figsize=(6, 6), dpi=80)
    plt.plot(lambda_list, rsme, marker='o')
    plt.margins(.10)
    plt.xlabel('$\lambda$')
    plt.ylabel('ERROR USING BEST $\\alpha$')
    plt.title('Figure 5')
    plt.text(.9, .20, "Widrow-Hoff", ha="center", va="center", rotation=0, size=15)
    plt.grid(True)
    plt.show()



if __name__ == '__main__':

    start_time = time.time()
    #figure_3()   #about --- 297.541052103 seconds ---
    #figure_4()  #about --- 126.288293123 seconds ---
    #figure_5()  #about --- 108.821047068 seconds ---
    print("--- %s seconds ---" % (time.time() - start_time))
