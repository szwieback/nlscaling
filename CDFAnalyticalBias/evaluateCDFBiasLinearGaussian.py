'''
evaluates the exact (no finite sample issues considered) bias of the Triple Correlation error variance \hat{sigma_x}^2
under the assumptions that the three data sets have been matched by CDF-matching and that all distributions are Gaussians

The stochastic signal model (zero mean) is assumed to be 
X=\alpha_x T + \epsilon_x
Y=\alpha_y T + \epsilon_y
Z=\alpha_z T + \epsilon_z
where T is the zero-mean anomaly (standard normal distribution) and \epsilon_i is Gaussian with variance \sigma_i^2, and the noise terms are uncorrelated.

The data sets are assumed to be rescaled, i.e. to have the same marginal normal distribution. This implies that
alpha_i^2 + \sigma_i^2 = D for all i

The bias is the expected value of the TC estimate \hat{sigma_x}^2 - \sigma_x^2; it is parameterized as a function of the alphas.
'''

def evaluateCDFBiasLinearGaussian(alphax,alphay,alphaz):
    b=alphax*(alphax-alphay-alphaz)+alphay*alphaz
    return b
