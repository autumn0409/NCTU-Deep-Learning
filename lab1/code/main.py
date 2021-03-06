from generate_data import generate_linear, generate_XOR_easy
from show_result import show_result
from utils import to_label, cal_accuracy
from net import FC_Net

epochs=50000
learning_rate=0.1

# train linear
net_linear = FC_Net(4, 4)
x_train, y_train = generate_linear()
linear_history = net_linear.train(x_train, y_train, epochs=epochs, learning_rate=learning_rate)

# test linear
y_hat = net_linear.test(x_train)
print('Testing output:')
print(y_hat)
y_hat_label = to_label(y_hat)
show_result(x_train, y_train, y_hat_label)
print(f'Accuracy = {cal_accuracy(y_train, y_hat)}')

# train XOR
net_XOR = FC_Net(10, 10)
x_train, y_train = generate_XOR_easy()
xor_history = net_XOR.train(x_train, y_train, epochs=epochs, learning_rate=learning_rate)

# test XOR
y_hat = net_XOR.test(x_train)
print('Testing output:')
print(y_hat)
y_hat_label = to_label(y_hat)
show_result(x_train, y_train, y_hat_label)
print(f'Accuracy = {cal_accuracy(y_train, y_hat)}')
