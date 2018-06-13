import csv
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    g = 1./(1+np.exp(-z))
    return g


def get_cost(X, y, theta):
    m = len(y)
    cost = (1 / m) * (-np.matmul(np.transpose(y), np.log(sigmoid(np.matmul(X, theta)))) -\
                      np.matmul(np.transpose(1 - y), np.log(1 - sigmoid(np.matmul(X, theta)))))
    return cost


def gradient(X, y, theta, e):
    m = len(y)
    gradient1 = ((1/m) * np.matmul(np.transpose(sigmoid(np.matmul(X, theta)) - y), X, ))
    gradient = np.zeros(shape=(e, 1))
    for i in range(0, e):
        gradient[i] = gradient1[0][i]
    return gradient


def gradient_descent(X, y, theta, alpha, k, e):
    costs = np.zeros(shape=(1, k))
    for j in range(0, k):
        theta -= alpha*gradient(X, y, theta, e)
        costs[0][j] = (get_cost(X, y, theta))
    return theta, costs


# read the csv
with open('train.csv', 'r')as csv_file:
    reader = csv.reader(csv_file, delimiter=',', quotechar='|')

    data = []

    for line in reader:
        data += ([line[1:]])

elements = 7

# making the array of data that we want
X = np.zeros(shape=(len(data)-1, elements))
y = np.zeros(shape=(len(data)-1, 1))


# making matrices that can be used to calculate things
for i in range(1, len(data)):
    y[(i - 1), 0] = data[i][0]  # survived
    X[(i - 1), 0] = 1           # have to put this in
    X[(i - 1), 1] = data[i][1]  # class
    if data[i][4] == 'male':    # sex
        X[(i - 1), 2] = 0
    else:
        X[(i - 1), 2] = 1
    if data[i][5] != '':        # Age
        X[(i - 1), 3] = data[i][5]
    X[(i - 1), 4] = data[i][6]  # sibling/spouse
    X[(i - 1), 5] = data[i][7]  # parent/child
    X[(i - 1), 6] = data[i][9]  # fare

initial_theta = np.zeros(shape=(elements, 1))

initial_cost = get_cost(X, y, initial_theta)

print('initial cost: ', initial_cost)

initial_gradient = gradient(X, y, initial_theta, elements)
print(initial_gradient)


alpha = .0035
iterations = 15000

final_theta, cost_history = gradient_descent(X, y, initial_theta, alpha, iterations, elements)
print(final_theta)
print(cost_history[0])
d = np.arange(0, iterations)
print(d)
plt.plot(d, cost_history[0], 'r-')
plt.show()


