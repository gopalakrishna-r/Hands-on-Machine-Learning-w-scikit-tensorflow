import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], 'ro-', label='Actual')
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], 'bx-', label='Forecast', markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)


def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", n_steps=50):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])


def plot_learning_curve(loss_, val_loss_):
    plt.plot(np.arange(len(loss_)) + 0.5, loss_, 'b.-', label ='Training loss')
    plt.plot(np.arange(len(val_loss_)) + 1, val_loss_, "r.-", label ='validation loss')
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer = True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize= 16)
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.show()
