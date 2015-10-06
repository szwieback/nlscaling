'''
Provides scenarios for testing the estimation of the parameters (error variances and mean map parameters, which correspond to generalized calibration constants)
For each, three estimation results are reported: 
    - triple collocation (assumes linear mean maps),
    - variational EM algorithm using the triple collocation results as initial values (assumes the 'correct' form of the mean maps, i.e. linear or quadratic)
    - variational EM algorithm using the actual values (in practice not known) as initial values (assumes the 'correct' form of the mean maps, i.e. linear or quadratic)
    - variational EM algorithm (linear)
    - triple collocation with linear orthogonal distance regression
    - triple collocation with nonlinear orthogonal distance regression
    - triple collocation with CDF matching
'''
import numpy as np
import GaussianBeliefNetwork as gbn
import GaussianRandomVariable as grv
import TimeSeriesDataFrame as tsdf
import TripleCollocation as tc
import TripleCollocationODR as tcODR
import TripleCollocationCDF as tcCDF
import copy
from pprint import pprint
def testEstimation(inputPGBN,seed0=1,size=1000,verbose=False,rvtype='Gaussian',**kwargs):
    # simulates time series based on inputPGBN (three children) and then estimates the parameters (mean maps, variances) by
    #    - triple collocation (linear mean maps)
    #    - variational EM (using the triple collocation and the actual parameters as starting values)
    # Input parameters:
    #    - inputPGBN, PyramidGaussianBeliefNetwork: probabilistic model of soil moisture products
    #    - seed0, integer (optional): set seed to seed0, if None: do not change seed
    #    - size, integer (optional): number of samples
    #    - verbose, boolean (optional): print summary of EM results
    #    - rvtype, string (optional): distribution to be used (Gaussian, Student, SkewNormal)
    #    - kwargs, dictionary (optional): additional parameters pertaining to the distributions    
    if seed0 is not None: np.random.seed(seed0)
    assert len(inputPGBN.children()) == 3
    
    try:
        outp={'variances':{},'meanmapparams':{}}    
    
        # preparation for output
        formats="{0:.5f}"
        variances=dict([(n.name,formats.format(n.variance)) for n in inputPGBN.children()])
        mmparams=dict([(n.name,[formats.format(i) for i in n.meanmaps[0].params]) for n in inputPGBN.children()])
        outp['variances']['actual']=dict([(n.name,n.variance) for n in inputPGBN.children()])
        outp['meanmapparams']['actual']=dict([(n.name,n.meanmaps[0].params) for n in inputPGBN.children()])
        
        # simulate time series and construct data frame
        inputPBN=assemble_PBN(inputPGBN,rvtype=rvtype,**kwargs)
        res=inputPBN.draw_from_joint(size=size)
        namelistv=[c.name for c in inputPGBN.children()]
        ts=tsdf.TimeSeriesDataFrame(namelistv,[res[n] for n in namelistv])
        
        # estimate by triple collocation
        tcPGBN=tc.TripleCollocation(ts)
        variancestc=dict([(n.name,formats.format(n.variance)) for n in tcPGBN.children()])
        mmparamstc=dict([(n.name,[formats.format(i) for i in n.meanmaps[0].params]) for n in tcPGBN.children()])
        outp['variances']['tc']=dict([(n.name,n.variance) for n in tcPGBN.children()])
        outp['meanmapparams']['tc']=dict([(n.name,n.meanmaps[0].params) for n in tcPGBN.children()])
        
        # linear ODR TC
        inputPGBNl=copy.deepcopy(tcPGBN)
        odrlPGBN=tcODR.TripleCollocationODR(ts,inputPGBNl,ts.names[0])
        outp['variances']['tcodrl']=dict([(n.name,n.variance) for n in odrlPGBN.children()])
        outp['meanmapparams']['tcodrl']=dict([(n.name,n.meanmaps[0].params) for n in odrlPGBN.children()])    
        
        # linear EM TC
        emtclPGBN=copy.deepcopy(tcPGBN)
        # re-initialize variance if negative
        for c in emtclPGBN.children():
            c.variance=np.abs(c.variance)
        mftcl=emtclPGBN.EM_Algorithm(ts,verbose=verbose,threshold=1e-4,lmax=50)
        outp['variances']['emtcl']=dict([(n.name,n.variance) for n in emtclPGBN.children()])
        outp['meanmapparams']['emtcl']=dict([(n.name,n.meanmaps[0].params) for n in emtclPGBN.children()])    
        
        # estimate the parameters by ODR TC
        odrPGBN=tcODR.TripleCollocationODR(ts,inputPGBN,'x')
        variancesodr=dict([(n.name,formats.format(n.variance)) for n in odrPGBN.children()])
        mmparamsodr=dict([(n.name,[formats.format(i) for i in n.meanmaps[0].params]) for n in odrPGBN.children()])
        outp['variances']['tcodr']=dict([(n.name,n.variance) for n in odrPGBN.children()])
        outp['meanmapparams']['tcodr']=dict([(n.name,n.meanmaps[0].params) for n in odrPGBN.children()])    
        
        # estimate by variational EM using TC results as initial values
        emtcPGBN=copy.deepcopy(odrPGBN)
        # re-initialize variance (often negative)
        for c in emtcPGBN.children():
            c.variance=np.abs(c.variance)
        mftcodr=emtcPGBN.EM_Algorithm(ts,verbose=verbose,threshold=1e-4,lmax=50)
        emtcPGBN2=copy.deepcopy(tcPGBN)
        # re-initialize variance (often negative) and correct mean map
        for c in emtcPGBN2.children():
            c.variance=np.abs(c.variance)
            if inputPGBN.get_random_variable(c.name).meanmaps[0].name == 'quadratic': c.meanmaps[0]=c.meanmaps[0].convert_to_quadratic()
        mftc=emtcPGBN2.EM_Algorithm(ts,verbose=verbose,threshold=1e-4,lmax=50)
        if mftc < mftcodr:
            emtcPGBN = emtcPGBN2
        variancesemtc=dict([(n.name,formats.format(n.variance)) for n in emtcPGBN.children()])
        mmparamsemtc=dict([(n.name,[formats.format(i) for i in n.meanmaps[0].params]) for n in emtcPGBN.children()])
        outp['variances']['emtc']=dict([(n.name,n.variance) for n in emtcPGBN.children()])
        outp['meanmapparams']['emtc']=dict([(n.name,n.meanmaps[0].params) for n in emtcPGBN.children()])
        
        # estimate the parameters by variational EM using actual values as initial values
        emPGBN=copy.deepcopy(inputPGBN)
        emPGBN.EM_Algorithm(ts,verbose=verbose)
        variancesem=dict([(n.name,formats.format(n.variance)) for n in emPGBN.children()])
        mmparamsem=dict([(n.name,[formats.format(i) for i in n.meanmaps[0].params]) for n in emPGBN.children()])
        outp['variances']['em']=dict([(n.name,n.variance) for n in emPGBN.children()])
        outp['meanmapparams']['em']=dict([(n.name,n.meanmaps[0].params) for n in emPGBN.children()])
        
        
        # estimate the variances by CDF matching (each individually)
        variancesCDFv=tcCDF.TripleCollocationCDF(ts)
        variancesCDF=dict([(n,formats.format(variancesCDFv[n])) for n in ts.names])
        mmparamsCDF=dict([(n,None) for n in ts.names])
        outp['variances']['tccdf']=variancesCDFv
        outp['meanmapparams']['tccdf']=mmparamsCDF
        
        if verbose:
            print 'Variances: actual'
            pprint(variances)
            print 'Variances: Triple Collocation'
            pprint(variancestc)
            print 'Variances: variational EM; inital values from Triple Collocation ODR'
            pprint(variancesemtc)
            print 'Variances: variational EM; actual values as initial values'
            pprint(variancesem)
            print 'Variances: Triple Collocation with ODR'
            pprint(variancesodr)
            print 'Variances: Triple Collocation with CDF'
            pprint(variancesCDF)
            print '\n'
            print 'MeanMap parameters: actual'
            pprint(mmparams)
            print 'MeanMap parameters: Triple Collocation'
            pprint(mmparamstc)
            print 'MeanMap parameters: variational EM; inital values from Triple Collocation ODR'
            pprint(mmparamsemtc)
            print 'MeanMap parameters: variational EM; actual values as initial values'
            pprint(mmparamsem)
            print 'MeanMap parameters: Triple Collocation with ODR'
            pprint(mmparamsodr)
            print 'MeanMap parameters: Triple Collocation with CDF'
            pprint(mmparamsCDF)
        return outp
    except:
        raise
        #return None
    
def assemble_PBN(inputPGBN,rvtype='Gaussian',**kwargs):
    if rvtype == 'Gaussian':
        return inputPGBN
    elif rvtype == 'Student':
        dof = kwargs['dof']
        rvtopnode=inputPGBN.topnode()
        rvtopnodem=grv.StudentRandomVariable(rvtopnode.name,[],[],rvtopnode.variance,dof)
        rvchildren=inputPGBN.children()
        rvchildrenm=[grv.StudentRandomVariable(rv.name,[rvtopnodem],rv.meanmaps,rv.variance,dof) for rv in rvchildren]
    elif rvtype == 'SkewNormal':
        skew = kwargs['skew']
        skewchildren = False
        if kwargs.has_key('skewchildren') and kwargs['skewchildren']: skewchildren = True
        rvtopnode=inputPGBN.topnode()
        rvtopnodem=grv.SkewNormalRandomVariable(rvtopnode.name,[],[],rvtopnode.variance,skew)
        rvchildren=inputPGBN.children()
        if skewchildren:
            skewch=skew
        else:
            skewch=0.
        rvchildrenm=[grv.SkewNormalRandomVariable(rv.name,[rvtopnodem],rv.meanmaps,rv.variance,skewch) for rv in rvchildren]
    elif rvtype == 'MixtureGaussian':
        weights=kwargs['weights']
        dmeans=kwargs['dmeans']
        variances=kwargs['variances']
        rvtopnode=inputPGBN.topnode()
        rvtopnodem=grv.MixtureGaussianRandomVariable(rvtopnode.name,[],[],weights,dmeans,variances)
        rvchildren=inputPGBN.children()
        rvchildrenm=[grv.GaussianRandomVariable(rv.name,[rvtopnodem],rv.meanmaps,rv.variance) for rv in rvchildren]
    else:
        raise NotImplementedError
    rvlistm=[rvtopnodem]+rvchildrenm
    return gbn.BeliefNetwork(rvlistm)    