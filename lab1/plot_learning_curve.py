import matplotlib.pyplot as plt

def plot_learning_curve(epochs, loss):
    plt.plot(epochs, loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')