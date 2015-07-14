Overview
****************************************

The nlscaling package comprises different methods for estimating error variances for soil moisture products that can be nonlinearly related

* **Triple Collocation** vanilla triple collocation (using instrumental variable rescaling)
* **Variational Expectation Maximization (EM) algorithm** based on a parametric probabilistic model (normal distributions)
* **Orthogonal Distance Regression** combined with triple collocation
* **Cumulative Distribution Function matching** combined with triple collocation

The probabilistic model underlying the variational EM describes both the unknown (latent) soil moisture and the observable products as random variables. A relevant question which arises both in the context of estimation using the variational EM algorithm and of product merging is that of inference of the underlying soil moisture given observations of (some) of the products, i.e. describing its probability distribution conditioned on the observations. When the relations (also called mean maps) between the products are nonlinear, this problem is difficult and approximate inference methods are required. Among those implemented are

* **Variational method** based on a Gaussian variational distribution as part of the EM algorithm: it minimizes a function of the discrepancy between the true (difficult to characterize) and the variational distribution
* **Markov Chain Monte Carlo** Metropolis sampling: draws samples from the conditional distribution
* **Linearization** based on linearized mean maps
* **Laplace method** also called saddle point method: it approximates the probability distribution around its mode
* **Quadrature** based on numerical integration for evaluating the posterior probability density