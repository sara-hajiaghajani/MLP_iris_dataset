
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('iris_dataset.csv')

# copy the data
normalize_dataset = dataset.copy()

#  normalization datasets
column = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
normalize_dataset[column] = (normalize_dataset[column] - normalize_dataset[column].min()) / (normalize_dataset[column].max() - normalize_dataset[column].min())


# bipolar targets
def bipolar_targets(df):
    T = []
    for i in range(len(df)):
        row = df.iloc[i]
        if row.type == "Iris-setosa":
            t1 = [1, -1, -1]
            T.append(t1)
        if row.type == "Iris-versicolor":
            t2 = [-1, 1, -1]
            T.append(t2)
        if row.type == "Iris-virginica":
            t3 = [-1, -1, 1]
            T.append(t3)
    df.insert(4, "bipolar_type", T)
    return df


# 80% train_data and 205 test-data
normalize_dataset = bipolar_targets(normalize_dataset)
normalize_dataset_train = normalize_dataset.sample(frac=0.8)
normalize_dataset_test = normalize_dataset.drop(normalize_dataset_train.index)
print(normalize_dataset)

# initialize n_inputs, n_outputs, n_hidden, train_inputs, test_inputs, train_outputs, test_outputs, beta, alfa
n_inputs = 4
n_outputs = 3
n_hidden = 7
train_inputs = normalize_dataset_train.iloc[:, [0, 1, 2, 3]]
train_outputs = normalize_dataset_train.iloc[:, [4]]
test_inputs = normalize_dataset_test.iloc[:, [0, 1, 2, 3]]
test_outputs = normalize_dataset_test.iloc[:, [4]]
# p = int(((n_inputs * n_hidden) + (n_hidden * n_outputs)) / 1.8)
beta = 0.7 * (pow(n_hidden, (1/4)))
alfa = 0.05


# Initialize random weights for start
def initialize_weights():
    # initialize inputs to hidden layer weights
    v_old = []
    v = []
    for i in range(n_inputs):
        for j in range(n_hidden):
            v_old.append(round(random.uniform(-0.5, 0.5), 2))
    v_old_norm = np.linalg.norm(v_old, 2)
    v_old_1 = [element * beta for element in v_old]
    for k in range(len(v_old_1)):
        v.append(round((v_old_1[k] / v_old_norm), 4))

    v_bias = []
    for j in range(n_hidden):
        v_bias.append(round(random.uniform(-beta, beta), 2))

    # initialize  hidden layer  to output weights
    w_old = []
    w = []
    for j in range(n_hidden):
        for z in range(n_outputs):
            w_old.append(round(random.uniform(-0.5, 0.5), 2))
    w_old_norm = np.linalg.norm(w_old, 2)
    w_old_1 = [element * beta for element in w_old]
    for k in range(len(w_old_1)):
        w.append(round((w_old_1[k] / w_old_norm), 4))


    w_bias = []
    for j in range(n_outputs):
        w_bias.append(round(random.uniform(-beta, beta), 2))
    return v, v_bias, w, w_bias


# Forward propagate
def forward_propagate(v, v_bias, inputs, w, w_bias):
    # determine hidden layer output
    z_in = []
    for i in range(n_hidden):
        z1 = 0
        for j in range(n_inputs):
            z1 += ((v[j + (n_inputs*i)]) * (inputs.iat[0, j]))
        z1 += v_bias[i]
        z_in.append(z1)
    # bipolar sigmoid activation function for hidden layer output
    z = []
    for i in range(len(z_in)):
        x = round(((-1) * z_in[i]), 2)
        z.append((1.0 - np.exp(x)) / (1.0 + np.exp(x)))
    # determine output layer output
    y_in = []
    for i in range(n_outputs):
        y1 = 0
        for j in range(n_hidden):
            y1 += (w[j + (n_hidden*i)]) * z[j]
        y1 += w_bias[i]
        y_in.append(y1)
    # bipolar sigmoid activation function for output layer output
    y = []
    for i in range(len(y_in)):
        x1 = round(((-1) * y_in[i]), 2)
        y.append(round(((1.0 - np.exp(x1)) / (1.0 + np.exp(x1))), 4))
    return z_in, z, y_in, y


# Backpropagation error
def Backpropagation_error(outputs, y, z, w, inputs):
    # first error factor for output layer
    s1 = []
    for i in range(len(y)):
        f_prim_y_in = (1 / 2) * (1.0 + y[i]) * (1.0 - y[i])
        x = outputs.iat[0, 0]
        s = (x[i] - y[i]) * f_prim_y_in
        s1.append(s)
    # calculate weight changes between hidden and output layer
    delta_w = []
    for i in range(len(s1)):
        for j in range(len(z)):
            h = alfa * s1[i] * z[j]
            delta_w.append(round(h, 4))
    # calculate bias weight changes between hidden and output layer
    delta_w_bias = []
    for i in range(len(s1)):
        delta_w_bias.append(round((alfa * s1[i]), 4))
    # second error factor for hidden layer
    s_in = []
    s2 = []
    for i in range(len(w)):
        k = 0
        for j in range(len(s1)):
            k += s1[j] * w[i]
        s_in.append(k)

    for j in range(len(z)):
        f_prim_z_in = (1 / 2) * (1.0 + z[j]) * (1.0 - z[j])
        s2.append(s_in[j] * f_prim_z_in)
    # calculate weight changes between input and hidden layer
    delta_v = []
    for i in range(len(s2)):
        for j in range(n_inputs):
            delta_v.append(round((alfa * s2[i] * (inputs.iat[0, j])), 4))
    # calculate bias weight changes between input and hidden layer
    delta_v_bias = []
    for i in range(len(s2)):
        delta_v_bias.append(round((alfa * s2[i]), 4))
    return s1, delta_w, delta_w_bias, s2, delta_v, delta_v_bias


# Update network weights with error
def update_weights(w, delta_w, w_bias,  delta_w_bias, v, delta_v, v_bias, delta_v_bias):
    # update_w_weights
    w_new = []
    for i in range(len(w)):
        w_new.append(round((w[i] + delta_w[i]), 4))
    # update_w_bias_weights
    w_new_bias = []
    for i in range(len(w_bias)):
        w_new_bias.append(round((w_bias[i] + delta_w_bias[i]), 4))
    # update_v_weights
    v_new = []
    for i in range(len(v)):
        v_new.append(round((v[i] + delta_v[i]), 4))
    # update_v_bias_weights
    v_new_bias = []
    for i in range(len(v_bias)):
        v_new_bias.append(round((v_bias[i] + delta_v_bias[i]), 4))
    return w_new, w_new_bias, v_new, v_new_bias


def correctly_predict(outputs, y, total):
    x = outputs.iat[0, 0]
    n = 0
    for j in range(len(x)):
        if x[j] == 1:
            n = j
    max_number = y[0]
    m = 0
    for k in range(len(y)):
        if y[k] > max_number:
            max_number = y[k]
            m = k
    if n == m:
        total += 1
    return total

def classification_accuracy(test_count_row, test_total):
    accuracy = (test_total * 100) / test_count_row
    print("accuracy of classification is:", round(accuracy, 2), "%")
    return accuracy

def train_set_error_rating(train_total, train_count_row):
    print("in train part the number of correctly predicted outputs is", train_total, "from", train_count_row, "inputs")
    train_error_rate = ((train_count_row - train_total) / train_count_row) * 100
    print("train_error_rate:", round(train_error_rate, 2), "%")
    return train_error_rate

def test_set_error_rating(test_total, test_count_row):
    print("in test part the number of correctly predicted outputs is", test_total, "from", test_count_row, "inputs")
    test_error_rate = ((test_count_row - test_total) / test_count_row) * 100
    print("test_error_rate:", round(test_error_rate, 2), "%")
    return test_error_rate


# Train network
def train_network():
    train_total = 0
    x = []
    x1 = []
    epoch1 = []
    epoch = 0
    train_error_rate = 100
    train_count_row = normalize_dataset_train.shape[0]
    v, v_bias, w, w_bias = initialize_weights()
    while train_error_rate > 10:
        epoch += 1
        if epoch > 50:
            break
        print("we are in training part epoch:", epoch)
        epoch1.append(epoch)
        # calculate train accuracy in each epoch for plot
        ac = (train_total * 100) / train_count_row
        x.append(ac)
        train_total = 0
        # calculate test accuracy in each epoch for plot
        test_count_row, test_total, u1 = test_network(w, w_bias, v, v_bias)
        ac1 = (test_total * 100) / test_count_row
        x1.append(ac1)
        for i in range(train_count_row):
            inputs = train_inputs.iloc[[i], [0, 1, 2, 3]]
            outputs = train_outputs.iloc[[i], [0]]
            z_in, z, y_in, y = forward_propagate(v, v_bias, inputs, w, w_bias)
            s1, delta_w, delta_w_bias, s2, delta_v, delta_v_bias = Backpropagation_error(outputs, y, z, w, inputs)
            w_new, w_new_bias, v_new, v_new_bias = update_weights(w, delta_w, w_bias, delta_w_bias, v, delta_v, v_bias, delta_v_bias)
            w = w_new
            w_bias = w_new_bias
            v = v_new
            v_bias = v_new_bias
            # calculate train error rate for while loop
            train_total = correctly_predict(outputs, y, train_total)
            train_error_rate = ((train_count_row - train_total) / train_count_row) * 100
    print("\n")
    print("final w=", w)
    print("final w_bias=", w_bias)
    print("final v=", v)
    print("final v_bias=", v_bias)
    return w, w_bias, v, v_bias, train_count_row, train_total, epoch, epoch1, x, x1


# Test network
def test_network(w, w_bias, v, v_bias):
    test_total = 0
    u = []
    test_count_row = normalize_dataset_test.shape[0]
    for i in range(test_count_row):
        inputs = test_inputs.iloc[[i], [0, 1, 2, 3]]
        outputs = train_outputs.iloc[[i], [0]]
        n = outputs.iat[0, 0]
        u.append("test output" + str(i+1) + ":")
        u.append(n)
        z_in, z, y_in, y = forward_propagate(v, v_bias, inputs, w, w_bias)
        u.append("y output" + str(i+1) + ":")
        u.append(y)
        # determining the accuracy for each input
        test_total = correctly_predict(outputs, y, test_total)
    return test_count_row, test_total, u


# result
w1, w_bias1, v1, v_bias1, count_row, total, epoch, epoch1, x, x1 = train_network()
count_row_1, total_1, u = test_network(w1, w_bias1, v1, v_bias1)
print("\n")
print("test result:", u)
print("\n")
print("learning rate:", alfa)
print("number of hidden layers:", n_hidden)
print("epoch:", epoch)
print("\n")
train_set_error_rating(total, count_row)
print("\n")
test_set_error_rating(total_1, count_row_1)
classification_accuracy(count_row_1, total_1)


# draw accuracy plot for train and test data in each epoch
plt.plot(epoch1, x, label='train accuracy')
plt.plot(epoch1, x1, label='test accuracy')
plt.title('model  accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy percent')
plt.legend()
plt.show()



