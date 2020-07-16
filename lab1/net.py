import numpy as np
from utils import cal_accuracy

class FC_Net:
    def __init__(self, hidden_dims=[10, 10]):
        input_width = 2
        output_width = 1
        self.hidden_layer_num = len(hidden_dims)
        
        # random generates weights
        self.W = []
        # weights between input layer and first hidden layer
        self.W.append(np.random.randn(input_width, hidden_dims[0]))
        # weights between hidden layers
        for l in range(1, self.hidden_layer_num):
            self.W.append(np.random.randn(hidden_dims[l - 1], hidden_dims[l]))
        # weights between last hidden layer and output layer
        self.W.append(np.random.randn(hidden_dims[-1], output_width))

    def train(self, x, y, epochs=100000, learning_rate=0.015):
        history = []
        print('Start training...')
        
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(x)
            self.backward(y, y_pred, learning_rate)
                
            loss = self.MSE(y, y_pred)
            acc = cal_accuracy(y, y_pred)
            history.append({'epoch': epoch, 'loss': loss, 'acc': acc})

            if (epoch % int(epochs / 50)) == 0:
                print(f'epoch {epoch:<6} loss: {loss:.5f}  accuracy: {acc:.5f}')

            if acc == 1.0:
                print(f'epoch {epoch:<6} loss: {loss:.5f}  accuracy: {acc:.5f}')
                print('Accuracy = 1.0, early stop the training.')
                break
                
        return history

    def test(self, x):
        return self.forward(x)

    def forward(self, x):
        self.z = []
        self.a = [x]

        for l in range(len(self.W)):
            self.z.append(np.matmul(self.a[l], self.W[l]))
            self.a.append(self.sigmoid(self.z[l]))

        return self.a[-1]

    def backward(self, y, y_pred, learning_rate):
        gradient_to_z = [None for l in range(len(self.W))]
        gradient_to_z[-1] = self.derivative_sigmoid(self.z[-1]) * self.derivative_MSE(y, y_pred)

        for l in range(len(self.W) - 1, -1, -1):
            if (l - 1) >= 0:
                gradient_to_z[l - 1] = np.matmul(gradient_to_z[l], self.W[l].T) * self.derivative_sigmoid(self.z[l - 1])

            # update weights
            gradient_to_w = np.matmul(self.a[l].T, gradient_to_z[l])
            self.W[l] = self.W[l] - learning_rate * gradient_to_w

    def MSE(self, y, y_pred):
        return np.square(np.subtract(y, y_pred)).mean()

    def derivative_MSE(self, y, y_pred):
        return -2 * (y - y_pred) / y.shape[0]

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_sigmoid(self, x):
        s = self.sigmoid(x)
        return np.multiply(s, 1.0 - s)

