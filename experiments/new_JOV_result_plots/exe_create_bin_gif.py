import subprocess
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from DLBio.helpers import check_mkdir, set_plt_font_size
from experiments.new_JOV_result_plots import exe_interactive_gradient as igrad
from experiments.new_JOV_result_plots import create_bin_plots as binp
from tqdm import tqdm

NUM_IMAGES = 50

BASE_FOLDER = 'experiments/new_JOV_result_plots'
IMAGE_FOLDER = join(BASE_FOLDER, 'gif_image')
FULL_GIF_PATH = '/nfshome/gruening/my_code/DLBio_repos/fp_nets_as_novel_deep_networks_inspired_by_vision/experiments/new_JOV_result_plots/gif_image'
check_mkdir(IMAGE_FOLDER)
set_plt_font_size(32)


def create_gamma_images():
    x_min = -4.
    x_max = +4.
    y_min = 0.
    y_max = 8.
    N = 1000
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N),
                         np.linspace(y_min, y_max, N))
    kwargs = {
        'x_min': -4.,
        'x_max': +4.,
        'y_min': 0.,
        'y_max': 8.,
        'xx': xx,
        'yy': yy,
    }

    gammas_ = np.linspace(1, 179, NUM_IMAGES)
    gammas_ = [int(g) for g in list(gammas_)]

    for name, gamma in tqdm(enumerate(gammas_)):
        plot_gamma_bin(gamma, **kwargs)
        plt.axis('off')
        save_fig(name, bbox_inches='tight')
        # return


def create_CIGA_images():
    # Does not work: 96x96 is too small to see something!
    eps = 1.
    tau = .001
    x0 = np.array([.01, .3]).reshape(1, 2)

    v = np.array([1., 0.]).reshape(1, 2)
    g = np.array([0., 1.]).reshape(1, 2)
    def f(x): return igrad.fp(v, g, x)
    def gf(x): return igrad.get_fp_grad(v, g, x)

    pos, _ = igrad.attack(f, gf, x0, eps, tau, mod=1, num_iterations=2000)
    indices = np.linspace(2, len(pos), NUM_IMAGES)
    indices = [int(i) for i in list(indices)]
    for name, idx in tqdm(enumerate(indices)):
        plot_ciga_grad(pos[:idx], f)
        save_fig(name)


def create_gif():
    # convert -delay 6 -resize 96x96 -loop 0 '*.png' 'output.gif'
    subprocess.call([
        'convert', '-delay', '6', '-resize', '96x96',
        '-loop', '0', '*.png', 'output.gif'],
        cwd=FULL_GIF_PATH
    )


def plot_gamma_bin(gamma0, *, xx, yy, x_min, y_min, x_max, y_max):
    Z = binp.get_fval_xy(xx, yy, gamma0)
    Z = Z.clip(min=0)
    Z = Z / Z.max()

    fig, ax = plt.subplots(1, figsize=(15, 15))
    plt.contour(yy, xx, Z, [0], colors='black', linewidths=3)

    Z = binp.bin_signal(Z, 6)
    im = ax.imshow(Z.T, extent=[y_min, y_max,
                                x_min, x_max], vmin=-1, vmax=1)
    plt.axis('equal')

    textstr = f'gamma: {int(gamma0)}°'
    props = dict(boxstyle='round', facecolor='white', alpha=0.75)
    # place a text box in upper left in axes coords
    ax.text(0.1, 0.95, textstr, transform=ax.transAxes, fontsize=60,
            verticalalignment='top', bbox=props)
    fig.set_size_inches(10, 10)


def plot_ciga_grad(pos, f):
    igrad.plot(pos, None, f, [1., 1.], 'v', 'g')
    plt.scatter(pos[-1, 0], pos[-1, 1], s=300, c='r', marker='d')


def save_fig(name, **kwargs):
    plt.savefig(join(IMAGE_FOLDER, str(name).zfill(3) + '.png'), **kwargs)
    plt.close()


if __name__ == '__main__':
    # create_CIGA_images()
    create_gamma_images()
    create_gif()
