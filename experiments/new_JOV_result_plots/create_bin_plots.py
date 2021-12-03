from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from DLBio.helpers import set_plt_font_size
from mpl_toolkits.axes_grid1 import make_axes_locatable

IM_OUT_FOLDER = 'experiments/new_JOV_result_plots/bin_plots'

set_plt_font_size(32)


def get_fval_xy(x, y, gamma0, lv=1., lg=1.):
    """Implementation of Equation 23

    Parameters
    ----------
    x : float array
        values for projection orthogonal to optimal stimulus
    y : float array
        values for projection along the optimal stimulus
    gamma0 : float
        angle value in degree
    lv : float, optional
        lenght of vector v, by default 1
    lg : float, optional
        lenght of vector g, by default 1

    Returns
    -------
    float array
        corresponding function value for each (x, y) pair
    """
    gamma = gamma0 / 180. * np.pi
    phi = .5 * gamma
    c1 = lv * lg

    return c1 * ((y * np.cos(phi))**2. - (x * np.sin(phi))**2.)


def bin_signal(x, num_bins=100):
    vmin = x.min()
    vmax = (x - vmin).max()

    # scale x to [0, 1]
    x -= vmin
    x /= vmax

    # prev. code returned 6 bins for num_bins=5
    num_bins -= 1
    # times nbins and round
    x = (x * num_bins).round()

    # rescale to [0, 1]
    x /= num_bins

    # rescale to [vmin, vmax]
    x = (x * vmax) + vmin

    return x


if __name__ == '__main__':
    x_min = -4.
    x_max = +4.

    y_min = 0.
    y_max = 8.

    N = 1000

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N),
                         np.linspace(y_min, y_max, N))

    for gamma0 in [0., 20., 45., 90., 135., 160.]:
        Z = get_fval_xy(xx, yy, gamma0)
        Z = Z.clip(min=0)
        Z = Z / Z.max()

        fig, ax = plt.subplots(1, figsize=(15, 15))
        plt.contour(yy, xx, Z, [0], colors='black', linewidths=3)

        Z = bin_signal(Z, 6)
        im = ax.imshow(Z.T, extent=[y_min, y_max,
                       x_min, x_max], vmin=-1, vmax=1)
        plt.axis('equal')

        textstr = f'gamma: {int(gamma0)}°'
        props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        # place a text box in upper left in axes coords

        if gamma0 == 0:
            # add a colorbar
            ax.text(0.45, 0.95, textstr, transform=ax.transAxes, fontsize=40,
                    verticalalignment='top', bbox=props)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.locator_params(nbins=3)

            fig.set_size_inches(10.56, 10)
        else:
            ax.text(0.45, 0.95, textstr, transform=ax.transAxes, fontsize=40,
                    verticalalignment='top', bbox=props)
            fig.set_size_inches(10, 10)

        plt.savefig(join(IM_OUT_FOLDER, f'bin_plot_gamma{int(gamma0)}.pdf'))
        plt.close()
