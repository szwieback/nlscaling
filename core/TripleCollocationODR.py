'''
TripleCollocationODR computes triple collocation variances and calibration constants and returns them as a PyramidGaussianBeliefNetwork
The underlying soil moisture variable is considered to be deterministic.
It can handle nonlinear polynomials in the mean maps in an approximate fashion.
The variances and calibration constants are computed iteratively: the calibration constants are obtained using orthogonal distance regressions (from scipy; in the paper it is referred to as total least squares), i.e. by considering the errors in both data sets.
The time series are subsequently 'inverted' (applying the inverse [as far as possible, the assumptions to make this a bijection are encoded in the used mean map]).
Subsequently, a 'bare' triple collocation is run on these 'rescaled' data, and the obtained variances are scaled back (using only the linear term); they form the new estimates for the next regression.
The starting values are obtained from a standard triple collocation.
'''
import scipy.odr as odr
import numpy as np
import copy
import TripleCollocation as tc
import TimeSeriesDataFrame as tsdf

def ODR(t,y,fun,variance_t,variance_y,beta0):# seems to be very sensitive to beta0
    # returns the polynomial coefficients beta by an orthogonal distance regression (wrapper to scipy routines)
    # input parameters
    #    - t, array: observed values of the explanatory variable t
    #    - y, array: observed values of the regressand y
    #    - fun, function: function f(beta, t) predicting y given the parameters beta and and t 
    #    - variance_t, float: variance of measurement of t
    #    - variance_y, float: variance of measurement of y
    #    - beta0, array: initial values of the parameters beta
    model=odr.Model(fun)
    data=odr.RealData(t,y,sx=np.sqrt(variance_t),sy=np.sqrt(variance_y))
    container=odr.ODR(data,model,beta0=beta0)
    outp=container.run()
    return outp.beta
 
def TripleCollocationODR(timeseriesdf,inputPGBN,childrefname,tolerance = 1e-6,maxsteps=10):
    # returns a PyramidGaussianBeliefNetwork containing the Triple Collocation results (obtained with orthogonal distance regression)
    # the regressions are done with respect to the product childrefname; the starting values are taken from a standard linear TC analysis
    # Input parameters:
    #    - timeseriesdf, TimeSeriesDataFrame: a data frame consisting of three time series
    #    - inputPGBN, PyramidGaussianBeliefNetwork: input PGBN that contains the structure (mean maps etc.)
    #    - childrefname, string: name of the reference product for the regressions (its mean map must be linear)
    #    - tolerance, float (optional): the iteration stops when the relative change of the error variances is less than this
    #    - maxsteps, integer (optional): the iteration stops after maxsteps cycles
   
    # check linearity of meanmap of childrefname
    assert inputPGBN.get_random_variable(childrefname).meanmaps[0].name == 'linear'
    
    # get starting values via standard triple collocation
    tcPGBN=tc.TripleCollocation(timeseriesdf)
    newPGBN=copy.deepcopy(tcPGBN)
    # re-initialize variance (often negative) to its absolute value and correct mean map
    for c in newPGBN.children():
        c.variance = np.abs(c.variance)
        if inputPGBN.get_random_variable(c.name).meanmaps[0].name == 'quadratic': c.meanmaps[0]=c.meanmaps[0].convert_to_quadratic()
    
    # set up vector of error variances (for stopping criterion)
    variancesvec=np.array([c.variance for c in newPGBN.children()])
    variancesvec0=np.array([0 for c in newPGBN.children()])
    iterw=0 # number of iterations
    # iteration: regression + scaling, triple collocation
    while np.linalg.norm((variancesvec-variancesvec0)/(variancesvec+variancesvec0))>tolerance and iterw<maxsteps:
        iterw+=1
        variancesvec0=variancesvec
        namelist=[childrefname]
        # scale back reference product
        mmref=newPGBN.get_random_variable(childrefname).meanmaps[0]
        valueslist=[mmref.invert(timeseriesdf.values_from_name(childrefname))]
        for c in newPGBN.children():
            if c.name != childrefname:
                # compute regression and scale back remaining products
                mmch=c.meanmaps[0]
                tx=(timeseriesdf.values_from_name(childrefname)-mmref.params[0])/mmref.params[1]
                y=timeseriesdf.values_from_name(c.name)
                beta0=np.asarray(mmch.params)
                # regression
                mmch.params=ODR(tx,y,mmch.get_fun_odr(),newPGBN.get_random_variable(childrefname).variance/mmref.params[1]**2,c.variance,beta0)
                namelist.append(c.name)
                # scale back
                valueslist.append(mmch.invert(y))
        
        # do bare TC        
        timeseriesdfrescaled=tsdf.TimeSeriesDataFrame(namelist,valueslist)        
        variancesrescaled=tc.TripleCollocationBare(timeseriesdfrescaled)
        #scale variances (linearly!)
        for c in newPGBN.children():
            c.variance=np.abs(variancesrescaled[c.name]*c.meanmaps[0].params[1]**2)
            
        #update mean map of reference node
        x=timeseriesdf.values_from_name(childrefname)
        alpha_x=np.mean(x)
        beta_x=np.sqrt(np.abs(np.var(x)-newPGBN.get_random_variable(childrefname).variance))
        newPGBN.get_random_variable(childrefname).meanmaps[0].params=[alpha_x,beta_x]
        variancesvec=np.array([c.variance for c in newPGBN.children()])

    return newPGBN
    