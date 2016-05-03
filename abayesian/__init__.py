#!/usr/bin/env python

import numpy as np
from scipy.special import beta
from scipy.special import betaln


class ABMeasure(object):
    """
    Helper class to tie together distribution tests.
    """

    def __init__(self):
        self.prob = 0
        self.converse = 1 - self.prob
        self.comparisons = {}


class BetaBinomial(ABMeasure):
    """
    Calculate statistical metrics for standard A/B tests using a beta
    distribution as a bayesian prior for a binomial distribution.
    """


    def compare(self, success_A, trials_A, success_B, trials_B):
        """
        Compute the probability that B > A given a beta distribution as a prior.

        Parameters
        -----------

        success_A: int
            Number of succesful trials for leg A of the test.

        trials_A: int
            Total number of trials on leg A of the test.

        success_B: int
            Number of succesful trials for leg B of the test.

        trials_B: int
            Total number of trials on leg B of the test.
        """

        self.prob = 0

        alpha_A = success_A + 1
        alpha_B = success_B + 1
        beta_A = (trials_A - success_A) + 1
        beta_B = (trials_B - success_B) + 1

        # If the alpha's are too large there will be a computational error.
        # This can be compensated for by computing the log beta (more costly)
        # instead.
        if alpha_B < 15 and alpha_A < 15:
            for i in np.arange(alpha_B):

                beta_1 = beta(alpha_A + i, beta_A + beta_B)
                beta_2 = beta(1 + i, beta_B) * beta(alpha_A, beta_A)
                self.prob += beta_1 / float((beta_B + i) * beta_2)

        else:
            for i in np.arange(alpha_B):

                beta_1 = betaln(alpha_A + i, beta_A + beta_B)
                beta_2 = betaln(1 + i, beta_B) + betaln(alpha_A, beta_A)
                self.prob += np.exp(beta_1 - np.log(beta_B + i) - beta_2)

        # Floating point errors can result in the probability being slightly
        # above 1.
        if self.prob > 1.:
            self.prob = 1.

        return self.prob


    def compare_many(self, test_success, test_trials, comparisons):
        """
        Compute the probability that B > A given a beta prior for many 
        different legs of the same A/B test.

        Parameters
        -----------

        test_success: int
            Number of succesful trials for testing leg.

        test_trials: int
            Number of trials for testing leg.

        comparisons : {test : [test_success, test_trials]}
            Dictionary of legs including name, success, and trials for each one
            of the legs.
        """

        self.prob = 0
        self.comparisons = {}

        for name, data in comparisons.iteritems():
            self.comparisons[name] = self.compare(data[0], data[1], test_success, test_trials)

        return self.comparisons


    def compare_effect(self, success_A, trials_A, success_B, trials_B, effect):
        """
        Compute the probability that B > A* given that A* is a multiple of a
        predefined leg A.

        Parameters
        -----------

        success_A: int
            Number of succesful trials for leg A of the test.

        trials_A: int
            Total number of trials on leg A of the test.

        success_B: int
            Number of succesful trials for leg B of the test.

        trials_B: int
            Total number of trials on leg B of the test.

        effect: float
            The desired effect size.
        """

        self.prob = 0
        success_A = int((1 + effect) * success_A)

        return self.compare(success_A, trials_A, success_B, trials_B)


    def trials_simple(self, success_rate, min_effect=None):
        """
        Given a certain success rate and desired effect range, naively compute
        the number of trials required to achieve an arbitrary significace.

        Parameters
        -----------

        success_rate: float
            The success rate for your test metric.

        min_effect: float
            Range of confidence interval determined by percent of success rate.
        """

        # If no minimum effect is specified, range it at 10 percent.
        if not min_effect:
            min_effect = .1

        stddev = success_rate * (1 - success_rate)
        self.trials = int(16. * stddev / float(np.power(stddev * min_effect, 2)))

        return self.trials


class PoissonGamma(ABMeasure):
    """
    Calculate statistical metrics for standard A/B tests using a gamma
    distribution as a bayesian prior for a poisson distribution.
    """


    def compare(self, arrivals_A, interval_A, arrivals_B, interval_B):
        """
        Compute the probability that B > A given a gamma distribution as a prior.

        Parameters
        -----------

        arrivals_A: int
            Number of arrivals for leg A of the test during an interval.

        interval_A: int
            Total number of intervals on leg A of the test.

        arrivals_B: int
            Number of arrivals for leg B of the test during an interval.

        interval_B: int
            Total number of intervals on leg B of the test.
        """
        
        self.prob = 0

        alpha_A = arrivals_A
        alpha_B = arrivals_B
        beta_A = interval_A
        beta_B = interval_B

        # Compute the probability that A > B.
        for i in np.arange(alpha_A):
            log_numer = (i * np.log(beta_A)) + (alpha_B * np.log(beta_B)) - ((alpha_B + i) * np.log(beta_B + beta_A))
            log_denom =  np.log(i + alpha_B) + betaln(i + 1, alpha_B)
            self.prob += np.exp(log_numer - log_denom)

        # Compute converse to determine B > A
        self.prob = 1 - self.prob

        # Floating point errors can result in the probability being slightly
        # above 1.
        if self.prob > 1.:
            self.prob = 1.

        return self.prob


    def compare_constant(self, arrivals_A, arrivals_B):
        """
        Compute the probability that B > A given a gamma distribution as a prior
        where both tests are on the same interval.

        Parameters
        -----------

        arrivals_A: int
            Number of arrivals for leg A of the test during an interval.

        arrivals_B: int
            Number of arrivals for leg B of the test during an interval.
        """

        return self.compare(arrivals_A, 1, arrivals_B, 1)



    def compare_many(self, arrivals, interval, comparisons):
        """
        Compute the probability that B > A given a gamma prior for many 
        different legs of the same A/B test.

        Parameters
        -----------

        arrivals: int
            Number of arrivals for testing leg.

        interval: int
            Interval where arrivals occurred for testing leg.

        comparisons : {test : [arrivals, interval]}
            Dictionary of legs including name, arrivals, and intervals for each 
            one of the legs.
        """

        self.prob = 0
        self.comparisons = {}

        for name, data in comparisons.iteritems():
            self.comparisons[name] = self.compare(data[0], data[1], arrivals, interval)

        return self.comparisons


    def compare_constant_many(self, arrivals, comparisons):
        """
        Compute the probability that B > A given a gamma prior for many 
        different legs of the same A/B test where both tests are on the 
        same interval.

        Parameters
        -----------

        arrivals: int
            Number of arrivals for testing leg.

        comparisons : {test : arrivals}
            Dictionary of legs including name, arrivals, and intervals for each 
            one of the legs.
        """

        self.prob = 0
        self.comparisons = {}

        for name, data in comparisons.iteritems():
            self.comparisons[name] = self.compare_constant(data, arrivals)

        return self.comparisons