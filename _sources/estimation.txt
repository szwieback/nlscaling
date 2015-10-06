========
Estimation of error variances
========

Overview
============================

The following code contains a function that applies all the different methods for error estimation to the *TimeSeriesDataFrame* ``ts``, which contains time series of the three products. For the parametric nonlinear methods, the product with the index 0 will have a linear mean map, whereas the others are taken to be quadratic functions.

The preamble imports relevant packages::

	import GaussianBeliefNetwork as gbn
	import GaussianRandomVariable as grv
	import TimeSeriesDataFrame as tsdf
	import TripleCollocation as tc
	import TripleCollocationODR as tcODR
	import TripleCollocationCDF as tcCDF
	import copy
	import numpy as np
	
The script only estimates errors when the timeseries ``ts`` (*TimeSeriesDataFrame*) are sufficiently long. It also prepares a dictionary ``outp`` that will be populated with the different estimates and ultimately returned::

	min_values = 100
	verbose = False
	if len(ts.values[0]) <= min_values:
		return None
	outp={'variances':{},'meanmapparams':{}}
		
Linear methods
============================

Subsequently vanilla triple collocation (TC) with instrumental variable rescaling is employed::

    # vanilla TC
    tcPGBN=tc.TripleCollocation(ts)
    outp['variances']['tc']=dict([(n.name,n.variance) for n in tcPGBN.children()])
    outp['meanmapparams']['tc']=dict([(n.name,n.meanmaps[0].params) for n in tcPGBN.children()])

Followed by TC using linear orthogonal distance regression (ODR), or total least-squares (TLS), rescaling::

    # linear ODR TC
    inputPGBNl=copy.deepcopy(tcPGBN)
    odrlPGBN=tcODR.TripleCollocationODR(ts,inputPGBNl,ts.names[0])
    outp['variances']['tcodrl']=dict([(n.name,n.variance) for n in odrlPGBN.children()])
    outp['meanmapparams']['tcodrl']=dict([(n.name,n.meanmaps[0].params) for n in odrlPGBN.children()])    
    
and linear error estimation based on variational EM algorithm (normality assumption)::

    # linear EM TC
    emtclPGBN=copy.deepcopy(tcPGBN)
    # re-initialize variance if negative
    for c in emtclPGBN.children():
        c.variance=np.abs(c.variance)
    mftcl=emtclPGBN.EM_Algorithm(ts,verbose=verbose,threshold=1e-4,lmax=50)
    outp['variances']['emtcl']=dict([(n.name,n.variance) for n in emtclPGBN.children()])
    outp['meanmapparams']['emtcl']=dict([(n.name,n.meanmaps[0].params) for n in emtclPGBN.children()])
	
Nonlinear methods
============================

The nonlinear methods all rely on the mean map parameterization encoded in the *PyramidGaussianBeliefNetwork* ``inputPGBN``, where all but the first (index 0) mean maps are set to quadratic functions::

    # generate PGBN with quadratic mean maps
    inputPGBN=copy.deepcopy(tcPGBN)    
    for c in inputPGBN.children():
        c.variance=np.abs(c.variance) # can be negative
        if c.name != ts.names[0]: c.meanmaps[0]=c.meanmaps[0].convert_to_quadratic() # covert all but first to quadratic

The nonlinear ODR method does not require initial values::
		
    # ODR TC
    odrPGBN=tcODR.TripleCollocationODR(ts,inputPGBN,ts.names[0])
    outp['variances']['tcodr']=dict([(n.name,n.variance) for n in odrPGBN.children()])
    outp['meanmapparams']['tcodr']=dict([(n.name,n.meanmaps[0].params) for n in odrPGBN.children()])    
    
while the variational EM algorithm does. These are taken from the results of the linear EM algorithm or the ODR regression; the version that achieves the minimum misfit (negative log likelihood) is subsequently adopted::
	
    # estimate by variational EM using TC results as initial values
    emtcPGBN=copy.deepcopy(odrPGBN)
    # re-initialize variance if negative
    for c in emtcPGBN.children():
        c.variance=np.abs(c.variance)
    mftcodr=emtcPGBN.EM_Algorithm(ts,verbose=verbose,threshold=1e-4,lmax=50)
    emtcPGBN2=copy.deepcopy(tcPGBN)
    # re-initialize variance if negative and correct mean map
    for c in emtcPGBN2.children():
        c.variance=np.abs(c.variance)
        if inputPGBN.get_random_variable(c.name).meanmaps[0].name == 'quadratic': c.meanmaps[0]=c.meanmaps[0].convert_to_quadratic()
    mftc=emtcPGBN2.EM_Algorithm(ts,verbose=verbose,threshold=1e-4,lmax=50)
    if mftc < mftcodr:
        emtcPGBN = emtcPGBN2
    outp['variances']['emtc']=dict([(n.name,n.variance) for n in emtcPGBN.children()])
    outp['meanmapparams']['emtc']=dict([(n.name,n.meanmaps[0].params) for n in emtcPGBN.children()])
    
The CDF matching needs no initialization::
	
    # estimate the variances by CDF matching (each individually)
    variancesCDFv=tcCDF.TripleCollocationCDF(ts)
    mmparamsCDF=dict([(n,None) for n in ts.names])
    outp['variances']['tccdf']=variancesCDFv
    outp['meanmapparams']['tccdf']=mmparamsCDF

