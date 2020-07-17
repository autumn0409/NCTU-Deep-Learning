import numpy as np
from utils import cal_accuracy

class FC_Net:
    def __init__(self, first_hidden_width, second_hidden_width):
        input_width = 2
        output_width = 1
        
        # random generates weights
        self.W = []
        self.W.append(np.random.randn(input_width, first_hidden_width))
        self.W.append(np.random.randn(first_hidden_width, second_hidden_width))
        self.W.append(np.random.randn(second_hidden_width, output_width))

    def train(self, x, y, epochs, learning_rate, early_stop=False):
        history = []
        print('Start training...')
        
        for epoch in range(1, epochs + 1):
            y_hat = self.forward(x)
            self.backward(y, y_hat, learning_rate)
                
            loss = self.MSE(y, y_hat)
            acc = cal_accuracy(y, y_hat)
            history.append({'epoch': epoch, 'loss': loss, 'acc': acc})

            if (epoch % int(epochs / 20)) == 0:
                print(f'epoch {epoch:<6} loss: {loss:.5f}  accuracy: {acc:.5f}')

            if early_stop and acc == 1.0:
                print(f'epoch {epoch:<6} loss: {loss:.5f}  accuracy: {acc:.5f}')
                print('Accuracy = 1.0, early stop the training.')
                break
                
        return history

    def test(self, x):
        return self.forward(x)

    def forward(self, x):
        self.z = []
        self.a = [x]

        for i in range(len(self.W)):
            self.z.append(np.matmul(self.a[i], self.W[i]))
            self.a.append(self.sigmoid(self.z[i]))

        return self.a[-1]

    def backward(self, y, y_hat, learning_rate):
        deltas = [None for i in range(len(self.W))]
        deltas[-1] = self.derivative_sigmoid(self.z[-1]) * self.derivative_MSE(y, y_hat)

        for i in range(len(self.W) - 1, -1, -1):
            if (i - 1) >= 0:
                deltas[i - 1] = np.matmul(deltas[i], self.W[i].T) * self.derivative_sigmoid(self.z[i - 1])

            # update weights
            gradient_to_w = np.matmul(self.a[i].T, deltas[i])
            self.W[i] -= learning_rate * gradient_to_w

    def MSE(self, y, y_hat):
        return np.square(np.subtract(y, y_hat)).mean()

    def derivative_MSE(self, y, y_hat):
        return -2 * (y - y_hat) / y.shape[0]

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_sigmoid(self, x):
        s = self.sigmoid(x)
        return np.multiply(s, 1.0 - s)

