"""
Tests to verify the equations in the JOV paper, especially the Section
'Iso-Response Contours'
"""
import numpy as np
import unittest

DIM = 9
SCALE = 10.
Z = .25


class EquationTests(unittest.TestCase):

    def test_gram_schmidt(self):
        """
        Is the vector computed by gram-schmidt really orthogonal?
        """
        for _ in range(100):
            v = self.get_random_vector()
            g = self.get_random_vector()
            o = FPNeuron.gram_schmidt(v, g)
            self.assertAlmostEqual(sp(v, o), 0.)

    def test_function_equivalence(self):
        """
        Are the functions F(x), F(x_ab) and f(a) equivalent? Eq. 7 & Eq. 21,
        also Eq. 15.
        """
        for _ in range(100):
            fp = self.create_random_fp_neuron()
            for _ in range(100):
                x = self.find_random_positive_x(fp)
                F_x = fp.F(x)

                a = fp.a(x)
                x_ab = fp.x_ab(a)
                F_x_ab = fp.F(x_ab)

                f_a = fp.f(a)

                self.assertAlmostEqual(F_x, F_x_ab)
                self.assertAlmostEqual(F_x, f_a)
                self.assertAlmostEqual(F_x_ab, f_a)

    def test_a_orth_and_a_opt(self):
        """
        Is a_orth orthogonal to the optimal stimulus a_opt?
        """
        for _ in range(100):
            fp = self.create_random_fp_neuron()
            a_opt = fp.a_opt()
            a_orth = fp.a_orth()
            self.assertAlmostEqual(sp(a_opt, a_orth), 0.)

    def test_iso_response(self):
        """
        Are all function values of (x,y)^T equal to Z? I.e., are all of these
        points on the iso-response curve?
        """
        for _ in range(100):
            fp = self.create_random_fp_neuron()
            a_opt = fp.a_opt()
            a_orth = fp.a_orth()
            for x in np.linspace(-2, 2, 1000):
                y = fp.y(x, Z)
                a = x * a_orth + y * a_opt

                x_ab = fp.x_ab(a)
                if not fp.is_positive_input(x_ab):
                    # the function (without ReLU) is symmetric
                    a = -1. * a

                self.assertAlmostEqual(fp.f(a), Z)

    def test_squared_descent(self):
        """
        When moving orthogonal to the optimal stimulus, does the function value
        decrease quadratically, as shown in Eq. 26?
        """
        for _ in range(100):
            fp = self.create_random_fp_neuron()
            a_opt = fp.a_opt()
            a_orth = fp.a_orth()
            for z in np.linspace(0.1, 10., 100):
                y0 = fp.y(0., z)

                x_ab = fp.x_ab(0 * a_orth + y0 * a_opt)
                if not fp.is_positive_input(x_ab):
                    # the function (without ReLU) is symmetric for both
                    # directions of y. If the input is negative, flip y.
                    y0 = -1 * y0

                f_00 = fp.f(0 * a_orth + y0 * a_opt)
                for x in np.linspace(-1., 1., 10):
                    delta = fp.delta_z(x)
                    true_delta = fp.f(x * a_orth + y0 * a_opt) - f_00
                    self.assertAlmostEqual(delta, true_delta)

    def create_random_fp_neuron(self):
        v = self.get_random_vector()
        g = self.get_random_vector()
        return FPNeuron(v, g)

    def find_random_positive_x(self, fp):
        """Find a vector x that has a positive scalar product with v and g.
        Since for the JOV-analysis, only these vectors were used.

        Parameters
        ----------
        fp : FPNeuron

        Returns
        -------
        np.array
            a vector x with sp(v,x) >= 0 and sp(g,x) >= 0

        Raises
        ------
        ValueError
            If no vector can be found an error is thrown
        """
        for _ in range(1000):
            x = self.get_random_vector()
            if fp.is_positive_input(x):
                return x

        raise ValueError('Could not find positive x')

    def get_random_vector(self):
        """Create a random uniform vector in [-SCALE/2, + SCALE/2]^DIM

        Returns
        -------
        np.array
        """
        return (np.random.rand(DIM) - .5) * SCALE


class FPNeuron():
    def __init__(self, v, g):
        """Constructor

        Parameters
        ----------
        v : np.array
            weight vector of size DIM
        g : np.array
            weight vector of size DIM
        """
        self.v = v
        self.g = g
        self.o = self.gram_schmidt(v, g)
        self.c1 = norm(v) * norm(g)
        self.gamma = compute_gamma(v, g)

    @staticmethod
    def gram_schmidt(v, g):
        """Return the part of g that is orthogonal to v

        Parameters
        ----------
        v : np.array
        g : np.array

        Returns
        -------
        np.array
            vector o with sp(v,o) = 0 and g = o + x*v
        """
        # eq 16
        return g - sp(v, g) / (norm(v)**2.) * v

    def is_positive_input(self, x):
        return sp(x, self.v) < 0 or sp(x, self.g) < 0

    def a(self, x):
        """Return the projection of x on the plane spanned by v and o

        Parameters
        ----------
        x : nD np.array

        Returns
        -------
        2D-np.array
        """
        return np.array([
            sp(x, self.v / norm(self.v)),
            sp(x, self.o / norm(self.o))
        ])

    def x_ab(self, a):
        """Inject the 2D-projection a back to the original nD-space. Note that
        there no vectors orthogonal to the v-g-plane in the output.

        Parameters
        ----------
        a : 2D np.array

        Returns
        -------
        nD np.array
            [description]
        """
        # eq 17
        assert a.shape[0] == 2
        return (
            a[0] / norm(self.v) * self.v + a[1] / norm(self.o) * self.o
        )

    def F(self, x):
        """Output of the simplified FP-neuron (Eq. 7) for nD inputs.

        Parameters
        ----------
        x : nD np.array

        Returns
        -------
        float
        """
        return sp(self.v, x) * sp(self.g, x)

    def f(self, a):
        """Output of the FP-neuron in projection space.

        Parameters
        ----------
        a : 2d np.array
            position on the v-g-plane

        Returns
        -------
        float
        """
        # eq 21
        assert a.shape[0] == 2
        h1 = np.array([
            a[0] * a[0],
            a[0] * a[1]
        ])
        h2 = self.c1 * np.array([
            np.cos(self.gamma),
            np.sin(self.gamma)
        ])

        return sp(h1, h2)

    def a_opt(self):
        """Direction of the optimal stimulus. On the v-g-plane.

        Returns
        -------
        2D np.array
        """
        return np.array([
            np.cos(self.gamma * .5),
            np.sin(self.gamma * .5),
        ])

    def a_orth(self):
        """A direction that is orthogonal to the optimal stimulus on the 
        v-g-plane.

        Returns
        -------
        2D np.array
        """
        return np.array([
            -1. * np.sin(self.gamma * .5),
            np.cos(self.gamma * .5),
        ])

    def y(self, x, z):
        """Given an iso-contour with the function value z and a position x
        along the axis a_orth: where is the position y along a_opt so that:

        f(x*a_orth + y*a_opt) = z.

        Parameters
        ----------
        x : float
            x-position on the iso-contour in the Frame (a_orth, a_opt)  
        z : float
            function value of the iso-contour

        Returns
        -------
        float
            y-position on the iso-contour in the Frame (a_orth, a_opt)  
        """
        # eq 24
        c1 = self.c1
        c = z / c1
        tan = np.tan(.5 * self.gamma)
        cos = np.cos(.5 * self.gamma)

        return np.sqrt(
            tan**2. * x**2. + c / cos**2.
        )

    def delta_z(self, x):
        """Compute the attenuation when moving away orthogonal to the optimal 
        stimulus (along a_orth). See Equation 26.

        Parameters
        ----------
        x : float
            How far do we move along a_orth

        Returns
        -------
        float
            Delta z = f(x* a_orth + y * a_opt) - z
        """
        return -1. * self.c1 * x**2. * np.sin(.5 * self.gamma)**2.


# ----------------------------------------------------------------------
# ------------------------ helpers -------------------------------------
# ----------------------------------------------------------------------


def sp(v, g):
    """Scalar product of the two vectors v and g

    Parameters
    ----------
    v : np.array
    g : np.array

    Returns
    -------
    float
    """
    return (v * g).sum()


def norm(x):
    return np.linalg.norm(x)


def compute_gamma(v, g):
    return np.arccos(
        sp(v, g) / (norm(v) * norm(g))
    )


if __name__ == '__main__':
    unittest.main()
