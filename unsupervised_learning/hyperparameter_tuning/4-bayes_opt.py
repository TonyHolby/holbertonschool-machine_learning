#!/usr/bin/env python3
"""
    A class named BayesianOptimization that performs Bayesian optimization
    on a noiseless 1D Gaussian process.
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
        A class named BayesianOptimization that performs Bayesian optimization
        on a noiseless 1D Gaussian process.
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
            Initializes Bayesian Optimization

            Args:
                f (float): the black-box function to be optimized.
                X_init (np.ndarray): a numpy.ndarray of shape (t, 1)
                    representing the inputs already sampled with the black-box
                    function.
                Y_init (np.ndarray): a numpy.ndarray of shape (t, 1)
                    representing the outputs of the black-box function for each
                    input in X_init:
                        t is the number of initial samples.
                bounds (tuple): a tuple of (min, max) representing the bounds
                    of the space in which to look for the optimal point.
                ac_samples (int): the number of samples that should be analyzed
                    during acquisition.
                l (int): the length parameter for the kernel.
                sigma_f (int): the standard deviation given to the output of
                    the black-box function.
                xsi (float): the exploration-exploitation factor for
                    acquisition.
                minimize (bool): a bool determining whether optimization should
                    be performed for minimization: True or maximization: False.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
            Calculates the next best sample location using the Expected
            Improvement acquisition function.

            Returns:
                X_next, EI:
                    X_next (np.ndarray): a numpy.ndarray of shape (1,)
                        representing the next best sample point.
                    EI (np.ndarray): a numpy.ndarray of shape (ac_samples,)
                        containing the expected improvement of each potential
                        sample.
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best = np.min(self.gp.Y)
            improvement = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            improvement = mu - best - self.xsi

        Z = np.zeros_like(improvement)
        EI = np.zeros_like(improvement)

        mask = sigma > 0
        Z[mask] = improvement[mask] / sigma[mask]
        EI[mask] = improvement[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])

        X_next = self.X_s[np.argmax(EI)]

        return X_next.reshape(1,), EI
