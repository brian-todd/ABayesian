'''
Hypothesis testing for Gaussian (Normal) distributions.
'''

import numpy as np

from typing import Optional

from numpy.linalg import solve
from scipy.stats import norm


class Gaussian():
    '''
    Calculate statistical metrics for standard A/B tests using a normal
    distribution.
    '''

    def compare(self, x_A: np.ndarray, x_B: np.ndarray) -> float:
        '''
        Compute the probability that B > A.

        Parameters
        -----------

        x_A: np.ndarray
            Raw data for leg A.

        x_B: np.ndarray
            Raw data for leg B.

        Returns
        -------

        prob: float
            The probability that one distribution is greater than the other.
        '''

        self.prob = 0.

        # Compute mean and variance for distribution.
        mean_A = x_A.mean()
        mean_B = x_B.mean()
        std_A = x_A.std()
        std_B = x_B.std()

        # Compute probability.
        mu = mean_B - mean_A
        sigma = np.sqrt(std_B + std_A)
        self.prob = 1 - norm.cdf(-1. * mu / float(sigma))

        # Correct floating point errors.
        if self.prob > 1.:
            self.prob = 1.

        return self.prob

class GaussianGaussian():
    '''
    Calculate statistical metrics for standard A/B tests using a normal
    distribution with a normal distribution as a prior.
    '''

    def compare(self, x_A, x_B, diffusive=1.):
        '''
        Compute the probability that B > A given a normal distribution as a prior.

        Parameters
        -----------

        x_A: np.ndarray
            Raw data for leg A.

        x_B: np.ndarray
            Raw data for leg B.

        diffusive: float
            Diffusive constant for determining variance in the prior.

        Returns
        -------

        prob: float
            The probability that one distribution is greater than the other.
        '''

        self.prob = 0.
        self.diffusive = diffusive

        std_A = x_A.std()
        std_B = x_B.std()

        # Derive prior hyperparameters.
        A = np.array([[1, 2], [1, -2]])
        b_A = np.array([x_A.max(), x_A.min()])
        b_B = np.array([x_B.max(), x_B.min()])

        mu_hyper_A, sigma_hyper_A = solve(A, b_A)
        mu_hyper_B, sigma_hyper_B = solve(A, b_B)

        # Force diffusive prior.
        sigma_hyper_A = self.diffusive * sigma_hyper_A
        sigma_hyper_B = self.diffusive * sigma_hyper_B

        # Derive parameters for posterior distribution.
        mu_number_A = mu_hyper_A / float(sigma_hyper_A ** 2) + x_A.sum() / float(std_A)
        mu_number_B = mu_hyper_B / float(sigma_hyper_B ** 2) + x_B.sum() / float(std_B)
        mu_denom_A = 1 / float(sigma_hyper_A ** 2) + len(x_A) / float(std_A)
        mu_denom_B = 1 / float(sigma_hyper_B ** 2) + len(x_B) / float(std_B)

        mu_A = mu_number_A / float(mu_denom_A)
        mu_B = mu_number_B / float(mu_denom_B)

        sigma_A = ((1 / sigma_hyper_A ** 2) + (len(x_A) / std_A)) ** -1.
        sigma_B = ((1 / sigma_hyper_B ** 2) + (len(x_B) / std_B)) ** -1.

        # Compute probability.
        mu = mu_B - mu_A
        sigma = np.sqrt(sigma_B ** 2 + sigma_A ** 2)
        self.prob = 1 - norm.cdf(-1. * mu / float(sigma))

        # Correct floating point errors.
        if self.prob > 1.:
            self.prob = 1.

        return self.prob
