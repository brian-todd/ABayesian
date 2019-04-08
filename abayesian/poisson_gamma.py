import numpy as np

from scipy.special import betaln


class PoissonGamma():
    """
    Calculate statistical metrics for standard A/B tests using a gamma
    distribution as a bayesian prior for a poisson distribution.
    """

    def compare(self, arrivals_A: int, interval_A: int, arrivals_B: int, interval_B: int) -> float:
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

        Returns
        -------

        prob: float
            The probability that one distribution is greater than the other.
        """

        self.prob = 0

        alpha_A = arrivals_A
        alpha_B = arrivals_B
        beta_A = interval_A
        beta_B = interval_B

        # Compute the probability that A > B.
        for i in np.arange(alpha_A):
            log_numer = (i * np.log(beta_A)) + (alpha_B * np.log(beta_B)) - ((alpha_B + i) * np.log(beta_B + beta_A))
            log_denom = np.log(i + alpha_B) + betaln(i + 1, alpha_B)
            self.prob += np.exp(log_numer - log_denom)

        # Compute converse to determine B > A
        self.prob = 1 - self.prob

        # Correct floating point errors.
        if self.prob > 1.:
            self.prob = 1.

        return self.prob

    def compare_constant(self, arrivals_A: int, arrivals_B: int) -> float:
        """
        Compute the probability that B > A given a gamma distribution as a prior
        where both tests are on the same interval.

        Parameters
        -----------

        arrivals_A: int
            Number of arrivals for leg A of the test during an interval.

        arrivals_B: int
            Number of arrivals for leg B of the test during an interval.

        Returns
        -------

        prob: float
            The probability that one distribution is greater than the other given
            that the two distributions are sampled over the same interval.
        """

        return self.compare(arrivals_A, 1, arrivals_B, 1)

    def compare_many(self, arrivals: int, interval: int, distributions: dict) -> dict:
        """
        Compute the probability that B > A given a gamma prior for many
        different legs of the same A/B test.

        Parameters
        -----------

        arrivals: int
            Number of arrivals for testing leg.

        interval: int
            Interval where arrivals occurred for testing leg.

        distributions : {test : [arrivals, interval]}
            Dictionary of legs including name, arrivals, and intervals for each
            one of the legs.

        Returns
        -------

        comparisons: dict
            Dictionary of comparisons between each test.
        """

        self.prob = 0
        self.comparisons = {}

        for name, data in distributions.iteritems():
            self.comparisons[name] = self.compare(data[0], data[1], arrivals, interval)

        return self.comparisons

    def compare_constant_many(self, arrivals: int, distributions: int) -> dict:
        """
        Compute the probability that B > A given a gamma prior for many
        different legs of the same A/B test where both tests are on the
        same interval.

        Parameters
        -----------

        arrivals: int
            Number of arrivals for testing leg.

        distributions : {test : arrivals}
            Dictionary of legs including name, arrivals, and intervals for each
            one of the legs.

        Returns
        -------

        comparisons: dict
            Dictionary of comparisons between each test given that the two
            distributions are sampled over the same interval.
        """

        self.prob = 0
        self.comparisons = {}

        for name, data in distributions.iteritems():
            self.comparisons[name] = self.compare_constant(data, arrivals)

        return self.comparisons
