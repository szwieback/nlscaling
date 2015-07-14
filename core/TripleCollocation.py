'''
Contains two functions:
TripleCollocation computes the triple collocation results of a TimeSeriesDataFrame and returns them as a PyramidGaussianBeliefNetwork
TripleCollocationBare does the same but for a scaled data set; it returns the variances as a dictionary
'''
import numpy as np
import GaussianBeliefNetwork as gbn
import GaussianRandomVariable as grv
import MeanMap as mm
def TripleCollocation(timeseriesdf):
    # computes triple collocation results and returns them as a PyramidGaussianBeliefNetwork
    # Input parameters:
    #    - timeseriesdf, TimeSeriesDataFrame: a data frame consisting of three time series
    assert len(timeseriesdf.names) == 3
    # extract time series as arrays
    y0=timeseriesdf.values[0]
    y1=timeseriesdf.values[1]
    y2=timeseriesdf.values[2]
    # normalize them by subtracting their mean
    y0p=y0-np.mean(y0)
    y1p=y1-np.mean(y1)
    y2p=y2-np.mean(y2)
    # compute scaling factors
    beta1=np.mean(y1p*y2p)/np.mean(y0p*y2p)
    beta2=np.mean(y1p*y2p)/np.mean(y0p*y1p)
    beta0t=np.std(y0)
    # compute offsets
    alpha1=np.mean(y1-beta1*y0)
    alpha2=np.mean(y2-beta2*y0)
    alpha0t=np.mean(y0)
    # compute scaling factors and offsets with respect to scaled time series 0
    beta1t=beta1*beta0t
    alpha1t=beta1*alpha0t+alpha1
    beta2t=beta2*beta0t
    alpha2t=beta2*alpha0t+alpha2
    # rescale
    y0s=(y0-alpha0t)/beta0t
    y1s=(y1-alpha1t)/beta1t
    y2s=(y2-alpha2t)/beta2t
    # compute variances of rescaled products
    var0s=np.mean((y0s-y1s)*(y0s-y2s))
    var1s=np.mean((y1s-y0s)*(y1s-y2s))
    var2s=np.mean((y2s-y1s)*(y2s-y0s))
    # compute variances of original products
    var0=var0s*(beta0t)**2
    var1=var1s*(beta1t)**2
    var2=var2s*(beta2t)**2
    # define random variables and joint distribution
    anom=grv.GaussianRandomVariable('t',[],[],1)
    meas0=grv.GaussianRandomVariable(timeseriesdf.names[0],[anom],[mm.LinearMeanMap([alpha0t,beta0t])],var0)
    meas1=grv.GaussianRandomVariable(timeseriesdf.names[1],[anom],[mm.LinearMeanMap([alpha1t,beta1t])],var1)
    meas2=grv.GaussianRandomVariable(timeseriesdf.names[2],[anom],[mm.LinearMeanMap([alpha2t,beta2t])],var2)    
    return gbn.PyramidGaussianBeliefNetwork(anom,[meas0,meas1,meas2])
def TripleCollocationBare(timeseriesdf):
    # computes triple collocation results (without scaling) and returns the variances as a dictionary
    # input parameters
    #    - timeseriesdf, TimeSeriesDataFrame: a data frame consisting of three time series
    assert len(timeseriesdf.names) == 3
    # extract time series as arrays
    y0=timeseriesdf.values[0]
    y1=timeseriesdf.values[1]
    y2=timeseriesdf.values[2]
    var0=np.mean((y0-y1)*(y0-y2))
    var1=np.mean((y1-y0)*(y1-y2))
    var2=np.mean((y2-y1)*(y2-y0))
    return dict(zip(timeseriesdf.names,[var0,var1,var2]))