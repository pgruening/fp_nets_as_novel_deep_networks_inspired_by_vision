import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from DLBio.helpers import set_plt_font_size
BASE_FOLDER = 'experiments/new_JOV_result_plots'


def get_fp_grad(v, g, x):
    """Returns the gradient of an fp-neuron at position x with the filter 
    vectors v and g

    In the paper, Equation 8 corresponds to this function.

    Parameters
    ----------
    v : np.array
        filter vector 1
    g : np.array
        filter vector 2
    x : np.array
        position

    Returns
    -------
    np.array
        gradient of the function
    """
    sp_vx = (v * x).sum(-1)
    sp_gx = (g * x).sum(-1)
    return v * sp_gx + g * sp_vx


def get_ln_grad(w, x):
    """Return the gradient of an LN-neuron at position x

    Parameters
    ----------
    w : np.array
        the neuron's weight vector
    x : np.array
        position
    Returns
    -------
    np.array
        the gradient is simply the weight vector
    """
    return w


def fp(v, g, x):
    """FP-neuron function, as described in Equation 7 in the Paper.

    Parameters
    ----------
    v : np.array
        filter vector 1
    g : np.array
        filter vector 2
    x : np.array
        position

    Returns
    -------
    float
        function value
    """
    sp_vx = (v * x).sum(-1)
    sp_gx = (g * x).sum(-1)
    return sp_vx * sp_gx


def ln(w, x):
    """Return the function value of an LN-neuron at position x

    Parameters
    ----------
    w : np.array
        The neuron's weight vector
    x : np.array
        Position
    Returns
    -------
    np.array
        returns the scalar product of w and x
    """
    return (w * x).sum(-1)


def attack(f, gf, x0, eps, tau, mod=100, num_iterations=10000):
    """The CIGA algorithm as described in Equation 6 in the paper.

    Parameters
    ----------
    f : function
        The respective neurons function, should return a float
    gf : function
        The gradient function. Should return an array
    x0 : np.array
        The original input, i.e., the starting position of the algorithm.
    eps : float
        +/- eps define the bounds of the maximum norm of the deviation from x0,
        called 'nu'
    tau : float
        step size of the algorithm

    Returns
    -------
    [type]
        [description]
    """
    nu = np.zeros(x0.shape)
    positions = []
    losses = []
    for i in range(num_iterations):
        q = nu + tau * gf(x0 + nu)
        nu = q.clip(min=-eps, max=eps)
        if i % mod == 0:
            positions.append(x0 + nu)
            # losses.append((f(x0 + nu) - f(x0)) / (f(x0) + 1.))
            losses.append(f(x0 + nu))

    positions = np.concatenate(positions, 0)
    return positions, losses


def plot(pos, loss, f, x_opt, xlabel, ylabel):
    """Plot the movement of the CIGA algorithm and the optimal stimulus and
    the iso-response contours.

    Parameters
    ----------
    pos : list of np.arrays
        CIGA positions for each iteration
    loss : list of float
        function values for each iteration
    f : function
        The respective neurons function, should return a float
    x_opt : np.array
        direction of the optimal stimulus
    xlabel : str
    ylabel : str
    """

    # get contour plot
    N = 200
    xx = np.linspace(-.1, 2., N)
    yy = np.linspace(-.1, 2., N)
    xx, yy = np.meshgrid(xx, yy)
    x = np.stack([xx.flatten(), yy.flatten()], -1)
    zz = f(x).reshape(N, N)

    _, ax = plt.subplots(1, 1, figsize=(15, 15))
    CS = ax.contour(xx, yy, zz, linewidths=5)
    ax.clabel(CS, inline=True, fontsize=28)

    # plot CIGA movement
    ax.plot(pos[:, 0], pos[:, 1], '-.', linewidth=8)

    # plot optimal stimulus
    ax.plot([0., x_opt[0]], [0., x_opt[1]], 'k-', linewidth=6)

    # make the plot a bit nicer
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')


def run():
    set_plt_font_size(32)

    # Setup
    eps = 1.
    tau = .001
    x0 = np.array([.01, .3]).reshape(1, 2)

    # LN-neuron
    w = np.array([1., 0.]).reshape(1, 2)
    def f(x): return ln(w, x)
    def gf(x): return get_ln_grad(w, x)

    pos, loss = attack(f, gf, x0, eps, tau)
    plot(pos, loss, f, [w[0, 0], w[0, 1]], 'w', 'o')
    plt.savefig(join(BASE_FOLDER, 'ln.png'))
    plt.savefig(join(BASE_FOLDER, 'ln.pdf'))

    # FP-neuron
    v = np.array([1., 0.]).reshape(1, 2)
    g = np.array([0., 1.]).reshape(1, 2)
    def f(x): return fp(v, g, x)
    def gf(x): return get_fp_grad(v, g, x)

    pos, loss = attack(f, gf, x0, eps, tau)
    plot(pos, loss, f, [1., 1.], 'v', 'g')
    plt.savefig(join(BASE_FOLDER, 'fp.png'))
    plt.savefig(join(BASE_FOLDER, 'fp.pdf'))


if __name__ == '__main__':
    run()
