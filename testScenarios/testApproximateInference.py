'''
Provides several scenarios in which the different methods to perform inference for the top node are compared
This means: given observed values of the children (e.g. soil moisture data sets), what is the estimated (conditional) distribution of top node (e.g. underlying soil moisture)
The graphical output of each consists of the intervals (either quantiles, or mean plus/minus standard deviation) of the distribution of the top node given the observed children

In the linear scenarios (i.e. when all meanmaps are linear) the linear, variational, quadrature and Laplace methods should be identical (apart from rounding/discretization errors) and the sampling distribution should converge to the same results as well
In the quadratic scenarios, only the sampling method and the quadrature method are 'correct' (sampling converges to the correct interval); the others are different approximations
'''
import GaussianBeliefNetwork as gbn
import GaussianRandomVariable as grv
import MeanMap as mm
import numpy as np
def testApproximateInference(intervalsize,inputPGBN,valuesdict,seed0=None):
    if seed0 is not None: np.random.seed(seed0)
    intervli=inputPGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='linear')
    print 'linear: '
    print intervli
    intervla=inputPGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='laplace')
    print 'Laplace: '
    print intervla
    intervva=inputPGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='variational')
    print 'variational: '
    print intervva
    intervqu=inputPGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='quadrature')
    print 'quadrature: '
    print intervqu
    intervsa=inputPGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='sampling',samplesize=1e4)
    print 'sampling: '
    print intervsa


def scenarioLinear1():
    print 'Scenario: Linear 1; three children; all mean maps are linear; all children observed'
    print 'Results: percentiles of conditional distribution of top node: [5%, 50%, 95%]'
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([-0.1,0.7])],3e-2)
    meas2=grv.GaussianRandomVariable('y',[anom],[mm.LinearMeanMap([0.8,1.2])],1e-1)
    meas3=grv.GaussianRandomVariable('z',[anom],[mm.LinearMeanMap([0.2,1.4])],6e-2)
    inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
    valuesdict={'x':1.6,'y':1.8,'z':1.7}
    testApproximateInference(0.1,inputPGBN,valuesdict,seed0=1)
def scenarioLinear2():
    print 'Scenario: Linear 2; three children; all mean maps are linear; all children observed'
    print 'Results: interval based on moments of conditional distribution of top node: [mean - stdev, mean, mean + stdev]'
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([-0.1,0.7])],3e-2)
    meas2=grv.GaussianRandomVariable('y',[anom],[mm.LinearMeanMap([0.8,1.2])],1e-1)
    meas3=grv.GaussianRandomVariable('z',[anom],[mm.LinearMeanMap([0.2,1.4])],6e-2)
    inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
    valuesdict={'x':1.3,'y':2.2,'z':1.7}
    testApproximateInference(False,inputPGBN,valuesdict,seed0=1)
def scenarioLinear3():
    print 'Scenario: Linear 3; three children; all mean maps are linear; two children observed'
    print 'Results: interval based on moments of conditional distribution of top node: [mean - stdev, mean, mean + stdev]'
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([-0.1,0.7])],3e-2)
    meas2=grv.GaussianRandomVariable('y',[anom],[mm.LinearMeanMap([0.8,1.2])],1e-1)
    meas3=grv.GaussianRandomVariable('z',[anom],[mm.LinearMeanMap([0.2,1.4])],6e-2)
    inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
    valuesdict={'x':1.3,'y':2.2}
    testApproximateInference(False,inputPGBN,valuesdict,seed0=1)
def scenarioQuadratic1():
    print 'Scenario: Quadratic 1; three children; two mean maps are quadratic, one linear; all children observed'
    print 'Results: percentiles of conditional distribution of top node: [5%, 50%, 95%]'
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([-0.1,0.8])],3e-2)
    meas2=grv.GaussianRandomVariable('y',[anom],[mm.QuadraticMeanMap([0.1,1.2,-0.15])],3e-2)
    meas3=grv.GaussianRandomVariable('z',[anom],[mm.QuadraticMeanMap([0.2,1.1,0.1])],6e-2)
    inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
    valuesdict={'x':2.1,'y':2.2,'z':2.7}
    testApproximateInference(0.1,inputPGBN,valuesdict,seed0=1)
def scenarioQuadratic2():
    print 'Scenario: Quadratic 2; three children; two mean maps are quadratic, one linear; all children observed'
    print 'Results: interval based on moments of conditional distribution of top node: [mean - stdev, mean, mean + stdev]'
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([-0.1,0.8])],3e-2)
    meas2=grv.GaussianRandomVariable('y',[anom],[mm.QuadraticMeanMap([0.1,1.2,-0.15])],3e-2)
    meas3=grv.GaussianRandomVariable('z',[anom],[mm.QuadraticMeanMap([0.2,1.1,0.1])],6e-2)
    inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
    valuesdict={'x':1.3,'y':2.2,'z':1.7}
    testApproximateInference(False,inputPGBN,valuesdict,seed0=1)
def scenarioQuadratic3():
    print 'Scenario: Quadratic 3; three children; two mean maps are quadratic, one linear; two children observed'
    print 'Results: interval based on moments of conditional distribution of top node: [mean - stdev, mean, mean + stdev]'
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([-0.1,0.8])],3e-2)
    meas2=grv.GaussianRandomVariable('y',[anom],[mm.QuadraticMeanMap([0.1,1.2,-0.15])],3e-2)
    meas3=grv.GaussianRandomVariable('z',[anom],[mm.QuadraticMeanMap([0.2,1.1,0.1])],6e-2)
    inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
    valuesdict={'x':1.3,'y':2.2}
    testApproximateInference(False,inputPGBN,valuesdict,seed0=1)