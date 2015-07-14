'''
Implementation of the single-variable Metropolis method - a Markov Chain Monte Carlo technique to sample from a given 
(not necessarily normalized) density function. The proposal density is a Gaussian with specified variance.  
'''
import numpy as np
def metropolis(fundensity,x0,size,proposalvar):
    # single-variable metropolis algorithm
    # inputs: 
    #    - fundensity, function pointer: one argument, returns the not necessarily scaled probability density
    #    - x0, float: sample value from which the chain is initialized
    #    - size, integer: length of the returned Markov chain
    #    - proposalvar, float: variance of the proposal density (Gaussian)
    # output:
    #    - samples, float array: Markov chain of samples

    # initialize
    samples=np.zeros(size)
    xprev=x0
    funprev=fundensity(xprev)
    # loop over all samples
    for i in np.arange(size):
        #propose new value
        xprop=np.random.randn()*np.sqrt(proposalvar)+xprev
        funprop=fundensity(xprop)
        alpha=funprop/funprev
        # (probabilistic) criterion whether to accept new value
        accept = np.random.random() < alpha
        # store and overwrite values
        if accept:
            samples[i]=xprop
            xprev=xprop
            funprev=funprop
        else:
            samples[i]=xprev
    return samples
