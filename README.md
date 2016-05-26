# ABayesian
Bayesian measurement of A/B Tests.

![image](https://imgs.xkcd.com/comics/frequentists_vs_bayesians.png)

### What does it do?
ABayesian provides an API for using Bayesian statistics to determine the probability that one leg of an A/B test is better than the other in the long run. You can measure the following types of experiments:

* Repeated Bernoulli trials (Beta-Binomial)
* Arrival data (Poisson-Gamma)
* Sequential, normally distributed data (Gaussian, Gaussian-Gaussian)

Generally speaking, Bayesian statistics are a safer way to test hypotheses and help compensate for sample size issues.

### How does it work?
Each leg of the A/B test is treated as is treated as it's own probability distribution, and a posterior distribution is calculated based on the input data. The final step is to determine the probability that one leg of the A/B test is greater than the other, which is representative of the long term probability that a leg was a winner.

### Examples
Each test is treated as a separate class:
```python
import abayesian

test1 = abayesian.BetaBinomial()
test2 = abayesian.PoissonGamma()
```
Both tests have a method that compares two tests:
```python
test1.compare(5, 100, 10, 100)
test2.compare(10, 2, 5, 1)
```
Each of the returned values is the probability that B > A.
