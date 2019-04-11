# ABayesian
Bayesian measurement of A/B Tests.

![image](https://imgs.xkcd.com/comics/frequentists_vs_bayesians.png)

### What does it do?
ABayesian provides an API for using Bayesian statistics to determine the probability that one leg of an A/B test is better than the other in the long run. You can measure the following types of experiments:

* Repeated Bernoulli trials (Beta-Binomial)
* Arrival data (Poisson-Gamma)
* Sequential, normally distributed data (Gaussian, Gaussian-Gaussian)

### How does it work?
For each leg of an A/B test parameters are fit to a posterior distribution based on a non-informative prior. This provides us with posterior probability distributions for each leg. The next step is computing the probability that one posterior distribution is greater than the other. [The derivation of the proceess in closed form can be found here, courtesy of Evan Miller.](http://www.evanmiller.org/bayesian-ab-testing.html)

### Examples
Each test is treated as a separate class:
```python
from abayesian.beta_binomial import BetaBinomial
from abayesian.poisson_gamma import PoissonGamma

test1 = abayesian.BetaBinomial()
test2 = abayesian.PoissonGamma()
```
Both tests have a method that compares two tests:
```python
test1.compare(5, 100, 10, 100)
test2.compare(10, 2, 5, 1)
```
Each of the returned values is the probability that B > A.

