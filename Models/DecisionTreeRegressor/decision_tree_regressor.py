import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

train_size = 150
data = np.genfromtxt('data_set.csv', delimiter=',', skip_header=1)
x_train = data[:train_size, 0]
y_train = data[:train_size, 1]
x_test = data[train_size:, 0]
y_test = data[train_size:, 1]

def get_split_score(nodes):
    return sum(np.sum(np.square(np.mean(y_train[i]) - y_train[i])) for i in nodes)

def get_error(y, y_pred, p):
    error = round(np.sqrt(np.mean(np.square(y - np.array(y_pred)))), 4)
    print('RMSE for max node {} is {}'.format(p, error))
    return error

def model(x, p):
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_splits = {}
    node_means = {}

    node_indices[0] = np.array(range(train_size))
    is_terminal[0] = False
    need_split[0] = True

    while True:
        split_nodes = [k for k, v in need_split.items() if v]

        if not split_nodes:
            break

        for node in split_nodes:
            data_indices = node_indices[node]
            need_split[node] = False
            node_means[node] = np.mean(y_train[data_indices])

            if len(data_indices) <= p:
                is_terminal[node] = True
            else:
                is_terminal[node] = False

                values = np.sort(np.unique(x_train[data_indices]))
                split_positions = (values[:-1] + values[1:]) / 2
                split_scores = [get_split_score((data_indices[x_train[data_indices] <= split], data_indices[x_train[data_indices] > split]))
                               for split in split_positions]

                best_score = min(split_scores)
                best_split = split_positions[split_scores.index(best_score)]
                node_splits[node] = best_split

                left_indices = data_indices[x_train[data_indices] <= best_split]
                node_indices[2 * node + 1] = left_indices
                is_terminal[2 * node + 1] = False
                need_split[2 * node + 1] = True

                right_indices = data_indices[x_train[data_indices] > best_split]
                node_indices[2 * node + 2] = right_indices
                is_terminal[2 * node + 2] = False
                need_split[2 * node + 2] = True

    y_pred = []
    for i in range(len(x)):
        index = 0
        while 1:
            if is_terminal[index] == True:
                y_pred.append(node_means[index])
                break
            else:
                if x[i] <= node_splits[index]:
                    index = index * 2 + 1
                else:
                    index = index * 2 + 2

    return np.array(y_pred)

errors = [get_error(y_test, model(x_test, p), p) for p in range(5, 51, 5)]
plt.scatter(range(5, 51, 5), errors)

x_plot = np.arange(min(data[:, 0]),max(data[:, 0]),0.001)
y_plot = model(x_plot, 25)

plt.scatter(x_train, y_train, color='b')
plt.scatter(x_test, y_test, color='r')
plt.plot(x_plot, y_plot)
