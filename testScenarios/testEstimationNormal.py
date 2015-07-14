'''
Provides scenarios for testing the estimation of the parameters (error variances and mean map parameters, which correspond to generalized calibration constants)
The simulated data are drawn from normal distributions, which conforms to the assumption of the EM estimation algorithm.
The results are described in testEstimation
'''
import GaussianBeliefNetwork as gbn
import GaussianRandomVariable as grv
import MeanMap as mm
import testEstimation
def scenarioEstimationLinear(verbose=False,seed0=1,size=350):
    # test a linear PGBN
    # Input parameters:
    #    - verbose, boolean (optional): print summary
    #    - seed0, integer (optional): seed to which the random number generator is set at the beginning
    #    - size, integer (optional): length of the simulated time series    
    print 'Scenario: Linear; three children; all mean maps are linear'
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([-0.1,0.7])],3e-2)
    meas2=grv.GaussianRandomVariable('y',[anom],[mm.LinearMeanMap([0.8,1.2])],1e-1)
    meas3=grv.GaussianRandomVariable('z',[anom],[mm.LinearMeanMap([0.2,1.4])],6e-2)
    inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
    return testEstimation.testEstimation(inputPGBN,seed0=seed0,size=size,verbose=verbose)
def scenarioEstimationQuadratic1(verbose=False,seed0=1,size=350):
    # test a quadratic (one linear) PGBN
    # Input parameters:
    #    - verbose, boolean (optional): print summary
    #    - seed0, integer (optional): seed to which the random number generator is set at the beginning
    #    - size, integer (optional): length of the simulated time series
    print 'Scenario: Quadratic 1; three children; one mean map is linear, the remaining two quadratic'
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([-0.1,0.8])],3e-2)
    meas2=grv.GaussianRandomVariable('y',[anom],[mm.QuadraticMeanMap([0.1,1.2,-0.15])],3e-2)
    meas3=grv.GaussianRandomVariable('z',[anom],[mm.QuadraticMeanMap([0.2,1.1,0.1])],6e-2)
    inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
    return testEstimation.testEstimation(inputPGBN,seed0=seed0,size=size,verbose=verbose)
def scenarioEstimationQuadratic2(verbose=False,seed0=1,size=350):
    # test a quadratic (one linear) PGBN
    # Input parameters:
    #    - verbose, boolean (optional): print summary
    #    - seed0, integer (optional): seed to which the random number generator is set at the beginning
    #    - size, integer (optional): length of the simulated time series
    print 'Scenario: Quadratic 2; three children; one mean map is linear, the remaining two quadratic'
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([0.3,0.1])],(0.04)**2)
    meas2=grv.GaussianRandomVariable('y',[anom],[mm.QuadraticMeanMap([0.2,0.05,0.005])],(0.08)**2)
    meas3=grv.GaussianRandomVariable('z',[anom],[mm.QuadraticMeanMap([0.25,0.06,-0.008])],(0.02)**2)
    inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
    return testEstimation.testEstimation(inputPGBN,seed0=seed0,size=size,verbose=verbose)
def scenarioEstimationQuadratic3(verbose=False,seed0=1,size=350):
    # test a quadratic (two linear) PGBN
    # Input parameters:
    #    - verbose, boolean (optional): print summary
    #    - seed0, integer (optional): seed to which the random number generator is set at the beginning
    #    - size, integer (optional): length of the simulated time series
    print 'Scenario: Quadratic 3; three children; two mean maps are linear, the remaining one quadratic'
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas1=grv.GaussianRandomVariable('x',[anom],[mm.LinearMeanMap([0.3,0.1])],(0.02)**2)
    meas2=grv.GaussianRandomVariable('y',[anom],[mm.LinearMeanMap([0.2,0.05])],(0.08)**2)
    meas3=grv.GaussianRandomVariable('z',[anom],[mm.QuadraticMeanMap([0.25,0.06,-0.01])],(0.05)**2)
    inputPGBN=gbn.PyramidGaussianBeliefNetwork(anom,[meas1,meas2,meas3])
    return testEstimation.testEstimation(inputPGBN,seed0=seed0,size=size,verbose=verbose)
