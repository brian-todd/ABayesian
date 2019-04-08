import numpy as np

from typing import Optional

from scipy.special import beta
from scipy.special import betaln


class BetaBinomial():
    '''
    Calculate statistical metrics for standard A/B tests using a beta
    distribution as a bayesian prior for a binomial distribution.
    '''

    def compare(self, success_A: int, trials_A: int, success_B: int, trials_B: int) -> float:
        '''
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

        Returns
        -------

        prob: float
            The probability that one distribution is greater than the other.
        '''

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

        # Correct floating point errors.
        if self.prob > 1.:
            self.prob = 1.

        return self.prob

    def compare_many(self, test_success: int, test_trials: int, distributions: dict) -> dict:
        '''
        Compute the probability that B > A given a beta prior for many
        different legs of the same A/B test.

        Parameters
        -----------

        test_success: int
            Number of succesful trials for testing leg.

        test_trials: int
            Number of trials for testing leg.

        distributions : {test : [test_success, test_trials]}
            Dictionary of legs including name, success, and trials for each one
            of the legs.

        Returns
        -------

        comparisons: dict
            Dictionary of comparisons between each test.
        '''

        self.prob = 0
        self.comparisons = {}

        for name, data in distributions.iteritems():
            self.comparisons[name] = self.compare(data[0], data[1], test_success, test_trials)

        return self.comparisons

    def compare_effect(self, success_A: int, trials_A: int, success_B: int, trials_B: int, effect: float) -> float:
        '''
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

        Returns
        -------

        prob: float
            The probability that one distribution is greater than the other
            given that we provide some "effect" modifier to the number of samples.
        '''

        self.prob = 0
        success_A = int((1 + effect) * success_A)

        return self.compare(success_A, trials_A, success_B, trials_B)

    def trials_simple(self, success_rate: float, min_effect: Optional[float] = None) -> int:
        '''
        Given a certain success rate and desired effect range, naively compute
        the number of trials required to achieve an arbitrary significace.

        NOTE: This method is a known, informal estimate. It is not a substitute
              benchmkaring significance.

        Parameters
        -----------

        success_rate: float
            The success rate for your test metric.

        min_effect: float
            Range of confidence interval determined by percent of success rate.

        Returns
        -------

        trials: int
            Estimated number of trials.
        '''

        # If no minimum effect is specified, range it at 10 percent.
        if not min_effect:
            min_effect = .1

        stddev = success_rate * (1 - success_rate)
        self.trials = int(16. * stddev / float(np.power(stddev * min_effect, 2)))

        return self.trials
