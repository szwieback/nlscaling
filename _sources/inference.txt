========
Approximate inference
========

Overview
============================

The following code is used for describing posterior probability intervals (unknown soil moisture given observations). The *PyramidGaussianBeliefNetwork* ``inputPGBN`` contains the structure and the dictionary ``valuesdict`` the scalar observations of the products. The variable ``intervalsize`` can be set to a quantile q (e.g. 0.1) so that the methods return the tuple of quantiles [q/2, 0.5, 1-q/2]; alternatively False yields [mean - stdev, mean, mean + stdev] of the posterior distribution::

	import GaussianBeliefNetwork as gbn
	import GaussianRandomVariable as grv
	import MeanMap as mm
	import numpy as np
	anom=grv.GaussianRandomVariable('t',[],[],1)
	meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([-0.1,0.8])],3e-2)
	meas2=grv.GaussianRandomVariable('y',[anom],[mm.QuadraticMeanMap([0.1,1.2,-0.15])],3e-2)
	meas3=grv.GaussianRandomVariable('z',[anom],[mm.QuadraticMeanMap([0.2,1.1,0.1])],6e-2)
	inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
	valuesdict={'x':1.3,'y':2.2,'z':1.7}
	
Methods
============================
	
Based on a linearization of the mean map::
	
	intervli=inputPGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='linear')
	print 'linear: '
	print intervli
	
Based on the Laplace approximation (Gaussian approximation around the posterior mode)::
	
    intervla=inputPGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='laplace')
    print 'Laplace: '
    print intervla
	
Based on a variational distribution (assumed Gaussian) by minimizing the Kullback-Leibler divergence, i.e. a function of the total difference of the unknown posterior and the variational distribution::
	
    intervva=inputPGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='variational')
    print 'variational: '
    print intervva

As opposed to the previous methods, which are all exact when the mean maps are linear, the approach based on quadrature (numerical evaluation of the posterior density function) is only approximate::

    intervqu=inputPGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='quadrature')
    print 'quadrature: '
    print intervqu

So is the sampling method (Metropolis Markov Chain Monte Carlo based on a Gaussian proposal distribution), drawing 10000 samples::
	
    intervsa=inputPGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='sampling',samplesize=1e4)
    print 'sampling: '
    print intervsa
	
