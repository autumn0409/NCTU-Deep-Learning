from generate_data import generate_linear, generate_XOR_easy
from show_result import show_result
from utils import to_label, cal_accuracy
from net import FC_Net

# train linear
net_linear = FC_Net([5, 5])
x_train, y_train = generate_linear()
net_linear.train(x_train, y_train, epochs=100000, learning_rate=0.015)

# show result
y_pred = net_linear.test(x_train)
print('Testing output:')
print(y_pred)
y_pred_label = to_label(y_pred)
show_result(x_train, y_train, y_pred_label)
print(f'Accuracy = {cal_accuracy(y_train, y_pred)}')

# train XOR
net_XOR = FC_Net([15, 15])
x_train, y_train = generate_XOR_easy()
net_XOR.train(x_train, y_train, epochs=100000, learning_rate=0.015)

# show result
y_pred = net_XOR.test(x_train)
print('Testing output:')
print(y_pred)
y_pred_label = to_label(y_pred)
show_result(x_train, y_train, y_pred_label)
print(f'Accuracy = {cal_accuracy(y_train, y_pred)}')

