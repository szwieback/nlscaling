'''
Provides the class GaussianRandomVariable, which is a normally distributed random variable whose conditional mean depends on the values of its parents via a MeanMap
This class is based upon the RandomVariable class, which is abstract and cannot be instantiated.

The daughter classes StudentRandomVariable, MixtureGaussianRandomVariable, and SkewNormalRandomVariable are only implemented in a rudimentary fashion.

The key methods of GaussianRandomVariable are:
    - drawing from the conditional distribution and evaluating the conditional density
    - evaluating certain terms in the variational bound of a GaussianBeliefNetwork
    - computing the M-Step and the necessary sufficient statistics for parameter testEstimation
'''

import numpy as np
from scipy.stats import truncnorm 
class RandomVariable(object):
    # abstract class that cannot be instantiated
    # provides several distribution-independent methods
    def __init__(self):
        raise Exception
    def parentsnames(self):
        # returns names of parents       
        return [p.name for p in self.parents]
    def conditional_mean(self,parentsvalues):
        # returns the conditional mean given the values of the parents (provided as list)
        # input parameters:
        #    - parentsvalues, list of floats: values of the parents
        if len(self.parentsnames())==0:
            condmean=0
        else:
            condmean=np.sum(np.array([m.evaluate(p) for m,p in zip(self.meanmaps,parentsvalues)]),axis=0)
        return condmean
    def unconditional_mean(self):
        # returns the unconditional mean
        if len(self.parentsnames())==0:
            uncondmean=0
        else:
            uncondmean=np.sum(np.array([m.evaluate(p) for m,p in zip(self.meanmaps,[p.unconditional_mean() for p in self.parents])]),axis=0)
        return uncondmean
    def _parentsinlist(self,rvlist):
        # checks if all the parents are in rvlist (returns Boolean)
        # input parameters:
        #    - rvlist, list of random variables: random variables to be checked
        return all((p in rvlist) for p in self.parents)

class StudentRandomVariable(RandomVariable):
    # only used for simulation
    # variance is the square of the scale parameter; the real variance is variance * dof / (dof-2) for dof > 2
    def __init__(self,name,parentslist,meanmapslist,spreadingpsquare,dof):
        # constructor method:
        # input parameters:
        #    - name, string: name of random variable
        #    - parentslist, list of random variables RandomVariable: the parents
        #    - meanmaplist, list of MeanMap: the mean maps in same order as in parentslist
        #    - spreadingpsquare, float: spreading parameter of t-distribution squared
        #    - dof, integer: degrees of freedom of t-distribution 
        self.variance=spreadingpsquare*(dof-2.0)/dof # the conditional variance
        self.name=name
        self.parents=parentslist
        self.meanmaps=meanmapslist
        self.dof=dof
    def draw_from_conditional_distribution(self,parentsvalues,size=1):
        # returns random samples from the conditional distribution given the values of the parents
        # input parameters:
        #    - parentsvalues, list of floats: values of the parents
        #    - size, integer (optional): number of i.i.d. samples drawn
        if len(parentsvalues) == 0:
            condmeans=np.zeros(size)
        else:
            condmeans=self.conditional_mean(parentsvalues)
        vals= np.random.standard_t(self.dof,size=len(condmeans))*np.sqrt(self.conditional_variance())+condmeans
        return vals
    def conditional_variance(self):
        # returns the conditional variance
        return self.variance

class TruncatedGaussianRandomVariable(RandomVariable):
    # only used for simulation, only implemented for 'orphan' nodes
    # conditional mean is the mean of the untruncated distribution!
    def __init__(self,name,parentslist,meanmapslist,variancenottruncated,lowerlimit,upperlimit):
        # constructor method:
        # input parameters:
        #    - name, string: name of random variable
        #    - parentslist, list of random variables RandomVariable: the parents
        #    - meanmaplist, list of MeanMap: the mean maps in same order as in parentslist
        #    - variancenottruncated, float: second conditional moment if truncation were not applied
        #    - lowerlimit, float: lower truncation limit in the 'space' of the standard normal distribution
        #    - upperlimit, float: upper truncation limit in the 'space' of the standard normal distribution
        self.name=name
        self.parents=parentslist
        self.meanmaps=meanmapslist
        self.upperlimit=upperlimit
        self.lowerlimit=lowerlimit
        self.variancenottruncated=variancenottruncated
    def draw_from_conditional_distribution(self,parentsvalues,size=1):
        # returns random samples from the conditional distribution given the values of the parents
        # input parameters:
        #    - parentsvalues, list of floats: values of the parents
        #    - size, integer (optional): number of i.i.d. samples drawn
        if len(parentsvalues) == 0:
            rvtemp=truncnorm(self.lowerlimit,self.upperlimit,loc=0,scale=np.sqrt(self.variancenottruncated))
            vals=rvtemp.rvs(size=size)
        else:
            raise NotImplementedError
        return vals
    def conditional_variance(self):
        # returns the conditional variance
        rvtemp=truncnorm(self.lowerlimit,self.upperlimit,loc=0,scale=np.sqrt(self.variancenottruncated))
        return rvtemp.moment(2)

class SkewNormalRandomVariable(RandomVariable):
    # only used for simulation
    # parameterized in terms of mean, variance and skew
    def __init__(self,name,parentslist,meanmapslist,variance,skew):
        # constructor method:
        # input parameters:
        #    - name, string: name of random variable
        #    - parentslist, list of random variables RandomVariable: the parents
        #    - meanmaplist, list of MeanMap: the mean maps in same order as in parentslist
        #    - variance, float: variance of conditional distribution
        #    - skew, float: skew of conditional distribution
        
        self.variance=variance # the conditional variance
        self.name=name
        self.parents=parentslist
        self.meanmaps=meanmapslist
        self.skew=skew
    def draw_from_conditional_distribution(self,parentsvalues,size=1):
        # returns random samples from the conditional distribution given the values of the parents
        # input parameters:
        #    - parentsvalues, list of floats: values of the parents
        #    - size, integer (optional): number of i.i.d. samples drawn
        if len(parentsvalues) == 0:
            condmeans=np.zeros(size)
        else:
            condmeans=self.conditional_mean(parentsvalues)
        import skew_normal
        vals=skew_normal.random_skewnormal(mean=condmeans,stdev=np.sqrt(self.variance),skew=self.skew,size=size) 
        return vals
    def conditional_variance(self):
        # returns the conditional variance
        return self.variance

class MixtureGaussianRandomVariable(RandomVariable):
    # only used for simulation
    # parameterized in terms of weights, dmeans, and variances of mixtures
    def __init__(self,name,parentslist,meanmapslist,weights,dmeans,variances):
        # constructor method:
        # input parameters:
        #    - name, string: name of random variable
        #    - parentslist, list of random variables RandomVariable: the parents
        #    - meanmaplist, list of MeanMap: the mean maps in same order as in parentslist
        #    - weights, list of float: weights of the mixture components; they sum to 1
        #    - dmeans, list of float: offsets (deviation of component mean from that provided by the mapping) of the individual components; their weighted sum should be 0
        #    - variances, float: variances of individual components
        assert np.abs(np.sum(np.array(weights))-1)<1e-6
        assert np.abs(np.sum(np.array(weights)*np.array(dmeans)))<1e-6
        self.variance=np.sum(np.array(weights)*(np.array(variances)+np.array(dmeans)**2)) # the conditional variance
        self.name=name
        self.parents=parentslist
        self.meanmaps=meanmapslist
        self.weights=weights
        self.dmeans=dmeans
        self.variances=variances
    def conditional_variance(self):
        # returns the conditional variance
        return self.variance
    def draw_from_conditional_distribution(self,parentsvalues,size=1):
        # returns random samples from the conditional distribution given the values of the parents
        # input parameters:
        #    - parentsvalues, list of floats: values of the parents
        #    - size, integer (optional): number of i.i.d. samples drawn
        if len(parentsvalues) == 0:
            condmeans=np.zeros(size)
        else:
            condmeans=self.conditional_mean(parentsvalues)
        # numpy trickery: first randomly determine the component by drawing from a uniform distribution and comparing to the cumulative weights
        cumw=np.cumsum(self.weights)
        mixturei=np.random.random_sample(size)        
        func1d=lambda x:np.nonzero(cumw > x)[0][0]
        vfunc=np.vectorize(func1d)
        ind=vfunc(mixturei)
        # draw from the mixture distribution by adding a Gaussian variable distributed according the corresponding mixture to condmeans
        admeans=np.array(self.dmeans)
        avariances=np.array(self.variances)
        vals= np.random.randn(size)*np.sqrt(avariances[ind])+admeans[ind]+condmeans
        return np.array(vals)    
    
class GaussianRandomVariable(RandomVariable):
    def __init__(self,name,parentslist,meanmapslist,variance):
        # constructor method:
        # input parameters:
        #    - name, string: name of random variable
        #    - parentslist, list of random variables RandomVariable: the parents
        #    - meanmaplist, list of MeanMap: the mean maps in same order as in parentslist
        #    - variance, float: the conditional variance
        self.variance=variance
        self.name=name
        self.parents=parentslist
        self.meanmaps=meanmapslist
    def conditional_variance(self):
        # returns the conditional variance
        return self.variance
    def draw_from_conditional_distribution(self,parentsvalues,size=1):
        # returns random samples from the conditional distribution given the values of the parents
        # input parameters:
        #    - parentsvalues, list of floats: values of the parents
        #    - size, integer (optional): number of i.i.d. samples drawn
        if len(parentsvalues) == 0:
            condmeans=np.zeros(size)
        else:
            condmeans=self.conditional_mean(parentsvalues)
        vals= np.random.randn(len(condmeans))*np.sqrt(self.conditional_variance())+condmeans
        return vals
    def get_conditional_log_density(self,value,parentsvalues):
        # returns the natural logarithm of the conditional density at value
        # input parameters:
        #    - value, float: value of random variable at which to evaluate
        #    - parentsvalues, list of floats: values of the parents
        m=self.conditional_mean(parentsvalues)
        p1=-0.5*np.log(2*np.pi*self.variance)
        p2=-((value-m)**2)/(2*self.variance)
        return p1+p2
    def sufficient_statistics_PGBN(self,value,varparamstuple):
        # returns a dictionary of one-instance sufficient statistics for the M-Step of the EM algorithm in parameter testEstimation in a PyramidGaussianBeliefNetwork
        # input parameters:
        #    - value, float: value of observed random variable
        #    - varparamstuple, tuple: float tuple as returned from the E-Step
        if len(self.meanmaps)==1: return self.meanmaps[0].sufficient_statistics_PGBN_child(value,varparamstuple)
        if len(self.meanmaps)==0: return {'e':np.exp(varparamstuple[1]),'mu':varparamstuple[0]}
    def F_var(self):
        # variance term in PGBN variational free energy
        return -0.5*np.log(2*np.pi*self.variance)
    def F_hidden(self,varparamstuple):
        # hidden variance term in PGBN variational free energy
        # input parameters:
        #    - varparamstuple, 2d-tuple: (variational) mean and logarithm of (variational) variance of random variable
        sigma0=np.sqrt(np.exp(varparamstuple[1]))
        return 0.5*(1+np.log(2*np.pi*sigma0**2)-(sigma0**2/self.variance))
    def F_mean_visible(self,value,parentsvarparamstuplelist):
        # child node term for PGBN variational free energy
        # input parameters:
        #    - value, float: observed value of the random variable
        #    - parentsvarparamstuplelist, list of 2d-tuple: list of (variational) means and logarithm of (variational) variances of parent
        assert len(parentsvarparamstuplelist) == 1 #>1 not implemented
        factor=-(2*self.variance)**(-1)
        n=(self.meanmaps[0]).F_n(parentsvarparamstuplelist[0])
        p1=(value-n)**2
        p2=(self.meanmaps[0]).F_quadform(parentsvarparamstuplelist[0])
        return factor*(p1+p2)
    def F_mean_top(self,varparamstuple):
        # top node term for PGBN variational free energy
        # input parameters:
        #    - varparamstuple, 2d-tuple: (variational) mean and logarithm of (variational) variance of random variable
        return (-(2*self.variance)**(-1))*varparamstuple[0]**2
    def M_step(self,suffstatnode,suffstatparent):
        # M-step for testEstimation of parameters of a child node in a PGBN
        # updates all the parameters
        # input parameters:
        #    -    suffstatnode, dictionary: sufficient statistics from sufficient_statistics_PGBN of the node
        #    -    suffstatparent, dictionary: sufficient statistics of only parent
        # output:
        #    -    metric, float: metric that measures the relative change of the parameter values
        dictsum={}
        
        # form sufficient statistics by averaging
        for k in suffstatnode.keys():
            L=len(suffstatnode[k])
            dictsum[k]=sum(suffstatnode[k])/L            
        dictsum['e']=sum(suffstatparent['e'])/L
        
        # compute MeanMap parameter estimates by solving system of linear equations 
        A=dictsum['a']+dictsum['b']
        b=dictsum['c']
        x=np.linalg.solve(A,b)
        
        # estimate conditional variance
        var=(dictsum['d']+np.dot(np.dot(x,dictsum['b']),x))
        
        # update parameter values
        params0=np.array(self.meanmaps[0].params)
        self.variance=var
        self.meanmaps[0].params=list(x)
        
        # compute metric of relative change
        metric=np.max(np.abs((x-params0)/(x+params0+1e-6)))
        return metric
        
''' 
# derivative terms, not needed for now and not tested
    def d_F_d_mean_mean_top(self,varparamstuple):
        sigma0=np.sqrt(np.exp(varparamstuple[1]))
        return -varparamstuple[0]/(sigma0**2)
    def d_F_d_mean_mean_visible(self,value,parentsvarparamstuplelist):
        n=(self.meanmaps[0]).F_n(parentsvarparamstuplelist[0])
        factor=(parentsvarparamstuplelist[0][0]-n)/self.variance
        p=(self.meanmaps[0]).d_F_mean_d_mean(parentsvarparamstuplelist[0])
        return factor*p
    def d_F_d_mean_variance_visible(self,value,parentsvarparamstuplelist):
        factor=-(2*self.variance)**(-1)
        p=(self.meanmaps[0]).d_F_var_d_mean(parentsvarparamstuplelist[0])
        return factor*p
    def d_F_d_var_var_top(self,varparamstuple):
        sigma0=np.sqrt(np.exp(varparamstuple[1]))
        return +0.5-(sigma0**2)/(2*self.variance)
    def d_F_d_var_mean_visible(self,value,parentsvarparamstuplelist):
        n=(self.meanmaps[0]).F_n(parentsvarparamstuplelist[0])
        factor=(parentsvarparamstuplelist[0][0]-n)/self.variance
        p=(self.meanmaps[0]).d_F_mean_d_var(parentsvarparamstuplelist[0])
        return factor*p
    def d_F_d_var_var_visible(self,value,parentsvarparamstuplelist):
        factor=-(2*self.variance)**(-1)
        p=(self.meanmaps[0]).d_F_var_d_var(parentsvarparamstuplelist[0])
        return factor*p
'''
