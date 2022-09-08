import numpy as np
import matplotlib.pyplot as plt
from os.path import join

BASE_FOLDER = 'thesis_plots'
P = np.array([.5, 0., 0.2])
T = 2.
SEED = 21
DEGREES = [1., 2., 10.]
NOISE = .05
NUM_POINTS = 10
V = 3.


def add_gaussian_noise(y, noise):
    return y + np.random.randn(*y.shape) * noise


def extrapolation():
    np.random.seed(SEED)
    x_train = np.linspace(-T, T, NUM_POINTS)
    delta = (x_train[1] - x_train[0]) * .5
    y_train = add_gaussian_noise(np.polyval(P, x_train), NOISE)

    x_test = np.concatenate([
        np.linspace(-V, -T - delta, 5),
        np.linspace(T + delta, V, 5),
    ])
    y_test = add_gaussian_noise(np.polyval(P, x_test), NOISE)

    all_data = np.concatenate([x_train, x_test])

    _, ax = plt.subplots(1, len(DEGREES), figsize=(12, 4))
    for i, deg in enumerate(DEGREES):
        fit = np.polyfit(x_train, y_train, deg)
        xp = np.linspace(-3., 3., 100)
        f_out = np.polyval(fit, xp)

        ax[i].plot(x_train, y_train, 'b.')
        ax[i].plot(x_test, y_test, 'k.')
        ax[i].plot(xp, f_out, '--', color='green')
        ax[i].grid()
        ax[i].set_ylim([-.5, 5.])
        ax[i].set_title(f'Degree: {int(deg)}')

    plt.tight_layout()
    plt.savefig(join(BASE_FOLDER, 'bias_var.png'))
    plt.savefig(join(BASE_FOLDER, 'bias_var.pdf'))


if __name__ == '__main__':
    extrapolation()
