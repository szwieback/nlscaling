'''
Contains the classes BeliefNetwork, GaussianBeliefNetwork and the inherited (less general) PyramidGaussianBeliefNetwork
BeliefNetwork is a rudimentary class that makes no assumptions about the random variables it includes
A GaussianBeliefNetwork is a Bayesian Network where all the conditional distributions are Gaussians, i.e. all nodes represent Gaussian Random Variables
The conditional means are related to the values of the parents by functional relations encoded by a MeanMap

A PyramidGaussianBeliefNetwork represents the special case where there is one top node of which all the remaining nodes are children; these are independent given the parent

The key methods of such a network are:
    - drawing random samples from the joint
    - evaluating the joint density
    - evaluating variational bounds
    - parameter testEstimation using the variational EM algorithm (only implemented for the case when only the topnode is hidden)
    - performing (approximate) inference (only implemented for topnode)
'''
import math,copy
import numpy as np
import GaussianRandomVariable as GRV
import MeanMap as MM
import scipy.optimize
import MCMCMetropolis
from scipy.misc import derivative
from scipy.integrate import quad
import scipy.stats
class BeliefNetwork(object):
    # base class for GaussianBeliefNetwork
    def __init__(self,rvlist):
        # Constructor
        # input parameters:
        #    - rvlist, list of RandomVariables: list of all the random variables (the structure is encoded in the parents)
        self.rvlist=rvlist
    def randomvariables(self,level=None,cumulative=True):
        # returns a list of all random variables
        # if provided, level (integer) gives only the variables at (cumulaitve == False) or below or including (cumulative == True) a certain level
        if level is None:
            return self.rvlist
        else:
            assert level>=0
            level=math.floor(level)
            if level == 0:
                rvs=[rv for rv in self.rvlist if len(rv.parentsnames())==0]
                return rvs
            else:# level>0
                rvsm1=self.randomvariables(level=level-1,cumulative=True)
                rvs=[rv for rv in self.rvlist if rv._parentsinlist(rvsm1)]
            if not cumulative:
                rvs=[rv for rv in rvs if rv not in rvsm1]
            return rvs
    def number_of_levels(self):
        # computes the number of levels of the network
        l=0
        while len(self.randomvariables(level=l,cumulative=False))>0:
            l=l+1
        return l
    def get_random_variable(self,rvname):
        # returns the random variable whose name is rvname (string)
        names=self.random_variable_names()
        return self.rvlist[names.index(rvname)]
    def random_variable_names(self):
        # returns a list of names of random variables (same order as in self.rvlist)
        return [r.name for r in self.rvlist]  

    def draw_from_joint(self,size=1):
        # draws samples from the joint distribution
        # input parameters:
        #    - size, float (optional): number of samples drawn
        # output:
        #    - outp, dictionary of arrays of values
        outp={}
        for l in np.arange(self.number_of_levels()):
            for rv in self.randomvariables(level=l,cumulative=False):
                parentsvalues=[outp[p.name] for p in rv.parents]
                values=rv.draw_from_conditional_distribution(parentsvalues,size=size)
                outp[rv.name]=values
        return (outp)
    def get_log_likelihood(self,valuesdict):
        # returns the log likelihood if all variables are observed
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        logl=0
        # loop over all variables (stratified by level) and add up the log densities
        for l in np.arange(self.number_of_levels()):
            for rv in self.randomvariables(level=l,cumulative=False):
                parentsvalues=[valuesdict[p.name] for p in rv.parents]
                logl+=rv.get_conditional_log_density(valuesdict[rv.name],parentsvalues)
        return logl

class GaussianBeliefNetwork(BeliefNetwork):
    def prune(self,rvnames):
        # returns a GBN with nodes whose names are stored in rvnames
        # input parameters:
        #    - rvnames, list of strings: names of rvnames to be kept
        return GaussianBeliefNetwork([self.get_random_variable(rvname) for rvname in rvnames])

class PyramidGaussianBeliefNetwork(GaussianBeliefNetwork):
    # a GBN with one variable at level 0, and n children of this top node
    def __init__(self,topGRV,childrenlist):
        # Constructor
        # input parameters:
        #    - topGRV, GaussianRandomVariable: top node
        #    - childrenlist, list of GaussianRandomVariables: list of the children
        self.rvlist=list(childrenlist)
        self.rvlist.insert(0, topGRV)
        assert self.number_of_levels()==2
    def topnode(self):
        # returns topnode
        return (self.rvlist[0])
    def children(self):
        # returns list of children
        return self.rvlist[1:]
    def linearize(self,valuesdict):
        # returns PyramidGaussianBeliefNetwork with linearized nodal random variables
        # linearization of the MeanMap occurs around the observed or, if hidden, the unconditional mean
        # input parameters:
        #    - valuesdict, dictionary: keys: names of observed random variables, values: their observed values
        visible=[k for (k,v) in valuesdict.iteritems() if v is not None]
        hidden=[rvname for rvname in self.random_variable_names() if rvname not in visible]
        if self.topnode().name in hidden:
            topnodevalue=self.topnode().unconditional_mean()
        else:
            topnodevalue=valuesdict[self.topnode().name]
        return self._linearize_all_observed(topnodevalue)
    def prune(self,rvnames):
        # returns a PGBN with nodes whose names are stored in rvnames; topnode is always kept
        # input parameters:
        #    - rvnames, list of strings: names of rvnames to be kept
        return PyramidGaussianBeliefNetwork(self.topnode(),[self.get_random_variable(rvname) for rvname in rvnames if rvname != self.topnode().name])
    def get_log_likelihood(self,valuesdict):
        # returns the log likelihood if all variables are observed (but can marginalize over unobserved children) 
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        logl=0
        for rvn in valuesdict.keys():
            rv=self.get_random_variable(rvn)
            parentsvalues=[valuesdict[p.name] for p in rv.parents]
            logl+=rv.get_conditional_log_density(valuesdict[rv.name],parentsvalues)
        return logl
    def _linearize_all_observed(self,topnodevalue):
        # helper method for linearize, assumes all random variables are observed
        topGRV=copy.deepcopy(self.rvlist[0])
        children=[GRV.GaussianRandomVariable(c.name,[topGRV],[mm.linearize(topnodevalue) for mm in c.meanmaps],c.variance) for c in self.children()]
        return PyramidGaussianBeliefNetwork(topGRV,children)
    def inference(self,valuesdict,linearizationthresh=1e-6,maxiter=100):
        # performs inference, i.e. computes the conditinal distribution of the unobserved variable (top node)
        # if the meanmaps are non-linear, they are linearized around current best guess until convergence
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        #    - linearizationthresh, float (optional): threshold for convergence (relative change)
        #    - maxiter, integer (optional): maximum number of iterations
        visible=[k for (k,v) in valuesdict.iteritems() if v is not None]
        # linearize around conditional means
        if self.topnode().name in visible:
            PTBNl=self.linearize(valuesdict)
            return PTBNl._inference_linear(valuesdict)
        else:
        # linearize around unconditional mean for topnode
            valuesdict2=copy.copy(valuesdict)
            topnodev=self.topnode().unconditional_mean()
            converged=False
            j=0
            while not converged and j<maxiter:
                PTBNl=self.linearize(valuesdict2)
                # perform linear inference
                meanch,Sigmachh,hiddenout = PTBNl._inference_linear(valuesdict)
                valuesdict2[self.topnode().name]=meanch[hiddenout.index(self.topnode().name)]
                if abs((topnodev-valuesdict2[self.topnode().name])/(valuesdict2[self.topnode().name]+topnodev+1e-6)) < linearizationthresh:
                    converged = True
                topnodev=valuesdict2[self.topnode().name]
                j=j+1
            return meanch,Sigmachh,hiddenout
    def _inference_linear(self,valuesdict):
        # helper method for inference (makes no restriction on which variables are observed)
        visible=[k for (k,v) in valuesdict.iteritems() if v is not None]
        hidden=[rvname for rvname in self.random_variable_names() if rvname not in visible]
        # preparations
        n=len(self.rvlist)
        indices_visible=[self.random_variable_names().index(rvn) for rvn in visible]
        indices_hidden=[self.random_variable_names().index(rvn) for rvn in hidden]
        P=self._permutation_matrix_for_inference(indices_hidden,indices_visible)
        meanP=np.dot(P,self.mean())
        a=np.array([valuesdict[rv] for rv in visible])
        covP=np.dot(np.dot(P,self.covariance()),P.T)
        # divide into hidden and visible vectors
        meanuh=meanP[0:len(hidden)]
        meanuv=meanP[len(hidden):n]
        Sigmauhv=covP[0:len(hidden),len(hidden):n]
        Sigmauvv=covP[len(hidden):n,len(hidden):n]
        Sigmauhh=covP[0:len(hidden),0:len(hidden)]
        # solve system of linear equations
        meanch=meanuh+np.dot(np.dot(Sigmauhv,np.linalg.inv(Sigmauvv)),(a-meanuv))
        Sigmachh=Sigmauhh-np.dot(np.dot(Sigmauhv,np.linalg.inv(Sigmauvv)),Sigmauhv.T)
        return meanch,Sigmachh,hidden
    def _laplace_topnode(self,valuesdict,stepsize=1e-4):
        # approximates the distribution of the hidden top node by the Laplace method
        # the density (not normalized) is locally (around the mode) approximated by a Gaussian
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        #    - stepsize, float (optional): step size for the numerical differentiation
        
        assert self.topnode().name not in valuesdict.keys()
        meanch,Sigmachh,hiddenout=self.inference(valuesdict)
        ind=hiddenout.index(self.topnode().name)
        t0=meanch[ind]
        # find mode
        funll=lambda t:-self.get_log_likelihood(dict(valuesdict.items()+{self.topnode().name:t}.items()))
        optmres=scipy.optimize.minimize(funll,t0)
        tout=optmres['x']
        # compute curvature
        varout=1/derivative(funll,tout,dx=stepsize,n=2) # check proportionality constant
        #print funll(t0)
        #print optmres['fun']
        #print optmres['success']
        #print optmres['nfev']
        return (tout[0],varout[0])
    def _quadrature_normalization_constant_topnode(self,valuesdict):
        # computes the partition function (normalization constant) of the conditional posterior of the topnode using quadrature
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
    
        funcondpnn=lambda t:np.exp(self.get_log_likelihood(dict(valuesdict.items()+{self.topnode().name:t}.items())))
        Z,err=quad(funcondpnn,-np.inf,np.inf)
        return Z
    def _quadrature_moment_topnode(self,valuesdict,moment,Z=None):
        # computes the specified moment of the conditional posterior of the topnode by quadrature
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        #    - moment, integer: moment to be computed (e.g. 1 for mean)
        #    - Z, float (optional): partition constant; if not provided, it will be computed using _quadrature_normalization_constant_topnode
        if Z is None: Z=self._quadrature_normalization_constant_topnode(valuesdict)
        funint=lambda t:(1/Z)*(t**moment)*np.exp(self.get_log_likelihood(dict(valuesdict.items()+{self.topnode().name:t}.items())))
        # numerical integration
        m,err=quad(funint,-np.inf,np.inf)
        return m
    def _quadrature_cdf_topnode(self,valuesdict,tval,Z=None):
        # computes the the cumulative distribution function of the posterior of the topnode (at tval)
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        #    - tval, float: value at which CDF is to be evaluated
        #    - Z, float (optional): partition constant; if not provided, it will be computed using _quadrature_normalization_constant_topnode
        if Z is None: Z=self._quadrature_normalization_constant_topnode(valuesdict)
        funint=lambda t:(1/Z)*np.exp(self.get_log_likelihood(dict(valuesdict.items()+{self.topnode().name:t}.items())))
        # numerical integration between -inf and tval
        P,err=quad(funint,-np.inf,tval)
        return P
    def _quadrature_inverse_cdf_topnode(self,valuesdict,P,Z=None):
        # computes the the inverse cumulative distribution function of the posterior of the topnode (at probability P)
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        #    - P, float: probability value at which the inverse CDF is to be evaluated
        #    - Z, float (optional): partition constant; if not provided, it will be computed using _quadrature_normalization_constant_topnode
        if Z is None: Z=self._quadrature_normalization_constant_topnode(valuesdict)
        # CDF(T)-P
        funcdfdev=lambda t:self._quadrature_cdf_topnode(valuesdict, t, Z=Z)-P
        # starting value: via normal approximation
        quadmean=self._quadrature_moment_topnode(valuesdict, 1, Z=Z)
        quadvar=self._quadrature_moment_topnode(valuesdict, 2, Z=Z)-quadmean**2
        t0=scipy.stats.norm.ppf(P,loc=quadmean,scale=np.sqrt(quadvar))
        # find point where CDF(t)=P, i.e. root of CDF(T)-P
        optmres=scipy.optimize.root(funcdfdev,t0)
        tout=optmres['x']
        return tout[0]
    def _sample_topnode(self,valuesdict,samplesize=1e4,proposalvar=1,burnin=100,thinningfactor=4):
        # computes a MCMC sample from the conditional distribution of the top node
        # it is obtained by thinning the chain of length samplesize by a factor thinningfactor, after having removed burnin samples from the beginning
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        #    - samplesize, integer (optional): length of Markov chain
        #    - proposalvar, float (optional): variance of the proposal distribution, which is a Gaussian
        #    - burnin, integer (optional): number of initial samples to discard (burnin period)
        #    - thinningfactor, integer (optional): take only every thinningfactor-th sample
        assert self.topnode().name not in valuesdict.keys()
        fundensity=lambda t:np.exp(self.get_log_likelihood(dict(valuesdict.items()+{self.topnode().name:t}.items())))
        samples=MCMCMetropolis.metropolis(fundensity,0,samplesize,proposalvar)
        samples=samples[np.arange(burnin,len(samples),thinningfactor)]
        return samples
    def mean(self):
        # returns the unconditional means as a vector of dimension n+1: top, children
        parentsvals=lambda x:[0 for y in range(len(x.parents))]
        return np.array([x.conditional_mean(parentsvals(x)) for x in self.rvlist])
    def covariance(self):
        # computes the unconditional covariance matrix
        # MeanMaps must be linear
        for rv in self.children():
            assert rv.meanmaps[0].name == 'linear'
        alphalist= [rv.meanmaps[0].params[1] for rv in self.children()]
        alphalist.insert(0,1)
        alphavec=np.array(alphalist)
        condvarvec=np.array([rv.conditional_variance() for rv in self.rvlist])
        condvarvec[0]=0
        return self.topnode().conditional_variance()*np.outer(alphavec,alphavec)+np.diag(condvarvec)
    def _permutation_matrix_for_inference(self,indices_visible,indices_hidden):
        # helper method which produces a permutation matrix to divide the indices into hidden and visible ones
        n=len(self.rvlist)
        indices=indices_visible+indices_hidden
        P=np.zeros([n,n])
        for i in np.arange(n):
            P[i,indices[i]]=1
        return P
    def variational_free_energy_one_instance(self,valuesdict,varparamstuple):
        # computes the variational free energy if all children are observed, top node unobserved
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        #    - varparamstuple, 2d-tuple: variational mean and logarithm of variational variance of top node 
        
        # check if the right variables are observed
        visible=[k for (k,v) in valuesdict.iteritems() if v is not None]
        hidden=[rvname for rvname in self.random_variable_names() if rvname not in visible]
        indices_visible=[self.random_variable_names().index(rvn) for rvn in visible]
        indices_hidden=[self.random_variable_names().index(rvn) for rvn in hidden]
        assert sorted(indices_visible) == range(1,len(self.children())+1)
        assert indices_hidden == [0]
        # collect the different terms
        F_var_top=self.topnode().F_var()
        F_var_children=np.sum([c.F_var() for c in self.children()])
        F_hidden_top=self.topnode().F_hidden(varparamstuple)
        F_mean_top=self.topnode().F_mean_top(varparamstuple)
        F_mean_children=np.sum([c.F_mean_visible(valuesdict[c.name],[varparamstuple]) for c in self.children()])
        return F_var_top+F_var_children+F_hidden_top+F_mean_top+F_mean_children
    def sufficient_statistics_PGBN(self,valuesdict,varparamstuple):
        # compute the one-instance sufficient statistics
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        #    - varparamstuple, 2d-tuple: variational mean and logarithm of variational variance of top node 
        # output:
        #    - suffstatdic, dictionary: keys: names of random variables, values: dictionary containing the different terms
        suffstatdic={}
        for c in self.children():
            suffstatdic[c.name]=c.sufficient_statistics_PGBN(valuesdict[c.name],varparamstuple)
        suffstatdic[self.topnode().name]=self.topnode().sufficient_statistics_PGBN(valuesdict,varparamstuple)
        return suffstatdic
    def E_Step_one_instance(self,valuesdict,varparamstuple0=None):
        # performs the variational E-Step (only top node hidden) given the observed values of one instance in valuesdict
        # if not provided, the initial values (varparamstuple0) are computed by linearization
        # input parameters:
        #    - valuesdict, dictionary: keys: names of random variables, values: their observed values
        #    - varparamstuple0, 2d-tuple (optional): initial values of variational mean and logarithm of variational variance of top node 
        # output:
        #    - Fout, float: value of the variational free energy after E-Step
        #    - suffstatsdict, dictionary: see sufficient_statistics_PGBN
        # initial values        
        if varparamstuple0 is None:
            meanch,Sigmachh,hiddenout=self.inference(valuesdict)
            ind=hiddenout.index(self.topnode().name)
            varparamstuple0=(meanch[ind],np.log(Sigmachh[ind,ind]))
        # optimization of F without analytical gradients
        #F=self.variational_free_energy_one_instance(valuesdict, varparamstuple0)        
        optmres=scipy.optimize.minimize(lambda x:-self.variational_free_energy_one_instance(valuesdict, (x[0],x[1])),varparamstuple0)#,lambda x:-np.array([self.d_variational_free_energy_one_instance_d_mean(valuesdict, (x[0],x[1])),self.d_variational_free_energy_one_instance_d_var(valuesdict, (x[0],x[1]))]))

        varparamstuple=optmres['x']
        
        Fout = optmres['fun']
        return (Fout,self.sufficient_statistics_PGBN(valuesdict,varparamstuple))

    def E_Step(self,timeseriesdf,varparams0=None):
        # performs the variational E-Step (only top node hidden) given the observed time series in timeseriesdf
        # if not provided, the initial values (varparamstuple0) are computed by linearization
        # input parameters:
        #    - timeseriesdf, TimeSeriesDataFrame: observed time series
        #    - varparamstuple0, list of 2d-tuple (optional): list of initial values of variational mean and logarithm of variational variance of top node 
        # output:
        #    - Fout, float: value of the total variational free energy after E-Step
        #    - suffstatsdict, dictionary: see sufficient_statistics_PGBN, but second level contains lists as these are the collected one-instance sufficient statistics
        
        suffstats=[]
        Fs=[]
        for i in range(timeseriesdf.get_length()): 
            if varparams0 is not None:
                varparams0tuple=varparams0[i]
            else:
                varparams0tuple=None
            (Fi,suffstatsi)=self.E_Step_one_instance(timeseriesdf.dictionary_value_instance(i),varparamstuple0=varparams0tuple)  
            suffstats.append(suffstatsi)
            Fs.append(Fi)
        suffstatsdict={}
        Ftotal=sum(Fs)
        for k in suffstats[0].keys():
            suffstatsdict[k]={}
            for kk in suffstats[0][k].keys():
                suffstatsdict[k][kk] = [d[k][kk] for d in suffstats]
        return (Ftotal,suffstatsdict)
    def M_Step(self,suffstatdict):
        # M-Step of variational EM algorithm
        # updates the relevant parameters
        # input parameters:
        #    - suffstatdict, dictionary of dictionary of lists: see E_Step
        # output:
        #    - metric, float: metric encoding the relative change of the parameters
        topnoden=self.topnode().name
        metrics={}
        for c in self.children():
            cn=c.name
            

            metrics[cn]=c.M_step(suffstatdict[cn],suffstatdict[topnoden])
        return max(metrics.values())
    def EM_Algorithm(self,timeseriesdf,threshold=1e-3,lmax=25,verbose=False):
        # performs the variational EM algorithm given the observed values n timeseriesdf
        # repeated until convergence (metric < threshold) or until lmax is reached
        # returns variational free energy
        # input parameters:
        #    - timeseriesdf, TimeSeriesDataFrame: observed values of the children
        #    - threshold, float (optional): threshold for metric (relative change in parameters)
        #    - lmax, integer (optional): maximum number of steps
        #    - verbose, boolean (optional): print summary of results
        metric=threshold+1
        varparams0=None
        l=0
        if verbose: print 'Starting EM Algorithm'
        # until maximum number of steps is reached or metric < threshold:
        while metric > threshold and l<lmax:
            # E-Step
            (Ftotal,suffstatdict)=self.E_Step(timeseriesdf,varparams0=varparams0)
            # set initial values for next E-Step
            varparams0=[(mu,np.log(nu2)) for mu,nu2 in zip(suffstatdict[self.topnode().name]['mu'],suffstatdict[self.topnode().name]['e'])]
            # M-Step
            metric=self.M_Step(suffstatdict)
            l=l+1
            if verbose: print l,Ftotal
        return Ftotal
    def interval_conditional_density_topnode(self,valuesdict,intervalsize,method='linear',**kwargs):
    # returns the bayesian probability interval (quantiles: 0.5*intervalsize, 0.5, 1-0.5*intervalsize) of the topnode conditioned on the children
    # if intervalsize is set to False: returns (mean - std dev, mean, mean + std dev); also conditioned on children
    # Input parameters:
    #    - valuesdict, dictionary: keys: names of observed random variables (children), values: their observed values
    #    - intervalsize, float: alpha value of probability interval (0 < intervalsize < 1)
    #    - method, string: linear, variational, laplace, sampling, quadrature
    #    - **kwargs: additional optional parameters: 
    #                - for linear: linearizationthresh; see inference method
    #                - for laplace: stepsize; see _laplace_topnode method
    #                - for variational: varparamstuple0; see E_Step_one_instance method
    #                - for sampling: samplesize, proposalvar, burnin, thinningfactor; see _sample_topnode method
    #                - for quadrature: -
        # get necessary information by calling method subroutines
        if method == 'linear':
            (mean,cov,names)=self.inference(valuesdict, **kwargs)
            mean=mean[names.index(self.topnode().name)]
            var=cov[names.index(self.topnode().name),names.index(self.topnode().name)]
            distribution='gaussian'
        elif method == 'laplace':
            (mean,var)=self._laplace_topnode(valuesdict, **kwargs)
            distribution='gaussian'
        elif method == 'variational':
            # prune network, i.e. remove unobserved children
            pPGBN=self.prune(valuesdict.keys())
            suffstat=pPGBN.E_Step_one_instance(valuesdict)[1]
            mean=suffstat[pPGBN.topnode().name]['mu']
            var=suffstat[pPGBN.topnode().name]['e']
            distribution='gaussian'
        elif method == 'sampling':
            samples=self._sample_topnode(valuesdict,**kwargs)
            distribution='empirical'
        elif method == 'quadrature':
            Z=self._quadrature_normalization_constant_topnode(valuesdict)
            distribution = 'exact'
            
        if intervalsize == False:
        # return (mean - std dev, mean, mean + std dev)
            if distribution == 'empirical':
                # empirical estimates of mean and variance
                mean=np.mean(samples)
                var=np.var(samples)
            elif distribution == 'exact':
                # quadrature estimates without any functional approximation
                mean=self._quadrature_moment_topnode(valuesdict, 1, Z)
                var=self._quadrature_moment_topnode(valuesdict, 2, Z) - mean**2
            std=np.sqrt(var)
            return np.array([mean-std,mean,mean+std])
        else:
        # return quantiles (0.5*intervalsize, 0.5, 1-0.5*intervalsize)
            if distribution == 'gaussian':
                std=np.sqrt(var)
                return np.array([scipy.stats.norm.ppf(0.5*intervalsize,loc=mean,scale=std),mean,scipy.stats.norm.ppf(1-0.5*intervalsize,loc=mean,scale=std)])
            elif distribution == 'empirical':
                # estimate from samples
                return np.array(np.percentile(samples,[50*intervalsize,50,100-50*intervalsize]))
            elif distribution == 'exact':
                # return quadrature estimates
                return np.array([self._quadrature_inverse_cdf_topnode(valuesdict, 0.5*intervalsize, Z=Z),self._quadrature_inverse_cdf_topnode(valuesdict, 0.5, Z=Z),self._quadrature_inverse_cdf_topnode(valuesdict, 1-0.5*intervalsize, Z=Z)])
        
'''
# not tested: partial derivatives of variational free energy 
    def d_variational_free_energy_one_instance_d_mean(self,valuesdict,varparamstuple):#dF/dmu_0 (top node)
        d_F_d_mean_mean_top=self.topnode().d_F_d_mean_mean_top(varparamstuple)
        d_F_d_mean_mean_children=np.sum([c.d_F_d_mean_mean_visible(valuesdict[c.name],[varparamstuple]) for c in self.children()])
        d_F_d_mean_variance_children=np.sum([c.d_F_d_mean_variance_visible(valuesdict[c.name],[varparamstuple]) for c in self.children()])
        return d_F_d_mean_mean_top+d_F_d_mean_mean_children+d_F_d_mean_variance_children
    def d_variational_free_energy_one_instance_d_var(self,valuesdict,varparamstuple):#dF/dlogsigma^2_0
        d_F_d_var_var_top=self.topnode().d_F_d_var_var_top(varparamstuple)
        d_F_d_var_mean_children=np.sum([c.d_F_d_var_mean_visible(valuesdict[c.name],[varparamstuple]) for c in self.children()])
        d_F_d_var_var_children=np.sum([c.d_F_d_var_var_visible(valuesdict[c.name],[varparamstuple]) for c in self.children()])
        return d_F_d_var_var_top+d_F_d_var_mean_children+d_F_d_var_var_children
'''