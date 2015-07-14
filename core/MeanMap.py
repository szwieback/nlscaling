'''
Provides the class MeanMap, which links the values of the parents to the conditional mean of their child
Contains two such maps (inherited classes from MeanMap): linear and quadratic

The key methods are:
    - evaluating the mapping function
    - evaluating certain terms that occur in the variational bounds of a GaussianBeliefNetwork
    - computing sufficient statistics for the EM algorithm
'''
import numpy as np
import copy
class MeanMap(object):
    def __init__(self,name,fun,params):
        # Constructor; mainly intended to be called from inherited class
        # input parameters:
        #    - name, string: name of the map
        #    - fun, function: takes t and params as input and returns map evaluated at t using parameters params
        #    - params, list: contains parameters to be used by function fun
        self.name=name
        self.fun=fun 
        self.params=params
    def evaluate(self,t):
        # maps t to the conditional mean f(t)
        # input parameters:
        #    - t, float: point at which fun is to be evaluated
        return self.fun(t,self.params)
    def get_fun_odr(self):
        # returns function that can be used by ODR
        return lambda params,t: self.fun(t,params)


class LinearMeanMap(MeanMap):
    # first order polynomial map
    def __init__(self,params):
        # Constructor, calls parent constructor
        # input parameters:
        #    - params, list of length 2: coefficients of first order polynomial: f(t)=params[0]+params[1]*t
        super(LinearMeanMap,self).__init__('linear',lambda x,y:y[0]+y[1]*x,params)
    def convert_to_quadratic(self):
        # returns a QuadraticMeanMap which encodes the same first-order polynomial
        return QuadraticMeanMap([self.params[0],self.params[1],0])
    def F_n(self,parentsvarparamstuple):
        # returns the output mean, if parent is a Gaussian with parameters encoded in parentsvarparamstuple
        # input parameters:
        #    - parentsvarparamstuple, 2d-tuple: (variational) mean and logarithm of (variational) variance of parent
        return self.evaluate(parentsvarparamstuple[0])
    def F_quadform(self,parentsvarparamstuple):
        # returns the output covariance matrix, if parent is a Gaussian with parameters encoded in parentsvarparamstuple
        # generalizes output variance of Frey & Hinton
        # input parameters:
        #    - parentsvarparamstuple, 2d-tuple: (variational) mean and logarithm of (variational) variance of parent
        sigma=np.sqrt(np.exp(parentsvarparamstuple[1]))
        C_11=sigma**2
        return (self.params[1]**2)*C_11
    def linearize(self,parentvalue):
        # returns a linearized version of itself, i.e. itself
        # input parameters:
        #    - parentvalue, float: value of the parent around which the map is linearized; not needed
        return LinearMeanMap(self.params)
    def sufficient_statistics_PGBN_child(self,value,parentsvarparamstuple):
        # computes the single-instance sufficient statistics of the observed variable given the observed value and the (variational) parameters of its parent
        # input parameters:
        #    - value, float: observed value of the child
        #    - parentsvarparamstuple, 2d-tuple: (variational) mean and logarithm of (variational) variance of parent
        sigma=np.sqrt(np.exp(parentsvarparamstuple[1]))
        m_0=1
        m_1=parentsvarparamstuple[0]
        m=np.array([m_0,m_1])
        C_11=sigma**2
        n=self.F_n(parentsvarparamstuple)
        a=np.outer(m,m)
        b=np.array([[0,0],[0,C_11]])
        c=value*m
        d=(value-n)**2
        return {'a':a,'b':b,'c':c,'d':d}
    def invert(self,values):
        # returns the parent values consistent with values
        # input parameters:
        #    - values, float array: observed values of the child
        return (values-self.params[0])/self.params[1]
'''
# derivatives of output mean/variance in variational bound; neither needed nor tested
    def d_F_mean_d_mean(self,parentsvarparamstuple):
        return self.params[1]
    def d_F_var_d_mean(self,parentsvarparamstuple):
        return 0
    def d_F_mean_d_var(self,parentsvarparamstuple):
        return 0
    def d_F_var_d_var(self,parentsvarparamstuple):
        sigma=np.sqrt(np.exp(parentsvarparamstuple[1]))
        d_C_11_d_var=sigma**2
        return (self.params[1]**2)*d_C_11_d_var
'''
class QuadraticMeanMap(MeanMap):
    # second order polynomial map
    def __init__(self,params):
        # Constructor, calls parent constructor
        # input parameters:
        #    - params, list of length 2: coefficients of first order polynomial: f(t)=params[0]+params[1]*t+params[2]*t*t
        super(QuadraticMeanMap,self).__init__('quadratic',lambda x,y:y[0]+y[1]*x+y[2]*(x*x),params)
    def linearize(self,parentvalue):
        # returns a linearized version of itself, i.e. itself
        # input parameters:
        #    - parentvalue, float: value of the parent around which the map is linearized
        u=parentvalue
        a=self.params[2]
        b=self.params[1]
        c=self.params[0]
        params2=[-a*u**2+c,2*a*u+b]
        return LinearMeanMap(params2)
    def F_n(self,parentsvarparamstuple):
        # returns the output mean, if parent is a Gaussian with parameters encoded in parentsvarparamstuple
        # input parameters:
        #    - parentsvarparamstuple, 2d-tuple: (variational) mean and logarithm of (variational) variance of parent
        sigma=np.sqrt(np.exp(parentsvarparamstuple[1]))
        M_const=1
        M_linear=parentsvarparamstuple[0]
        M_square=parentsvarparamstuple[0]**2+sigma**2
        return self.params[0]*M_const+self.params[1]*M_linear+self.params[2]*M_square
    def F_quadform(self,parentsvarparamstuple):
        # returns the output covariance matrix, if parent is a Gaussian with parameters encoded in parentsvarparamstuple
        # generalizes output variance of Frey & Hinton
        # input parameters:
        #    - parentsvarparamstuple, 2d-tuple: (variational) mean and logarithm of (variational) variance of parent
        sigma=np.sqrt(np.exp(parentsvarparamstuple[1]))
        mu=parentsvarparamstuple[0]
        C_11=sigma**2
        C_22=4*(sigma**2)*(mu**2)+sigma**4
        C_12=2*(sigma**2)*mu
        C=np.array([[0,0,0],[0,C_11,C_12],[0,C_12,C_22]])
        weights=np.array(self.params)
        res=np.dot(np.dot(weights,C),weights)
        return res
    def convert_to_quadratic(self):
        # returns a QuadraticMeanMap, i.e. a copy of itself
        return copy.deepcopy(self)
    def sufficient_statistics_PGBN_child(self,value,parentsvarparamstuple):
        # computes the single-instance sufficient statistics of the observed variable given the observed value and the (variational) parameters of its parent
        # input parameters:
        #    - value, float: observed value of the child
        #    - parentsvarparamstuple, 2d-tuple: (variational) mean and logarithm of (variational) variance of parent
        sigma=np.sqrt(np.exp(parentsvarparamstuple[1]))
        mu=parentsvarparamstuple[0]
        C_11=sigma**2
        C_22=4*(sigma**2)*(mu**2)+sigma**4
        C_12=2*(sigma**2)*mu
        m=np.array([1,mu,mu**2+sigma**2])
        n=self.F_n(parentsvarparamstuple)
        a=np.outer(m,m)
        b=np.array([[0,0,0],[0,C_11,C_12],[0,C_12,C_22]])
        c=value*m
        d=(value-n)**2
        return {'a':a,'b':b,'c':c,'d':d}
    def invert(self,values):
        # returns the parent values consistent with values [closer to 0]
        # input parameters:
        #    - values, float array: observed values of the child
        valuesparent=np.zeros_like(values)
        for i,v in enumerate(values):
            coeff=np.flipud(copy.copy(self.params))
            coeff[2]=coeff[2]-v
            polroots=np.roots(coeff)
            valuesparent[i]=np.real(polroots[np.argmin(np.abs(polroots))])
        return valuesparent
'''
# derivatives of output mean/variance in variational bound; neither needed nor tested
    def d_F_mean_d_mean(self,parentsvarparamstuple):
        p1=self.params[1]
        p2=self.params[2]*2*parentsvarparamstuple[0]
        return p1+p2
    def d_F_var_d_mean(self,parentsvarparamstuple):
        sigma=np.sqrt(np.exp(parentsvarparamstuple[1]))
        mu=parentsvarparamstuple[0]
        d_C11_d_mean=0
        d_C22_d_mean=8*mu*sigma**2
        d_C12_d_mean=2*sigma**2
        return d_C11_d_mean*(self.params[1]**2) + d_C22_d_mean*(self.params[2]**2) + 2*d_C12_d_mean*(self.params[1]*self.params[2])
    def d_F_mean_d_var(self,parentsvarparamstuple):
        sigma=np.sqrt(np.exp(parentsvarparamstuple[1]))
        return self.params[2]*(sigma**2)        
    def d_F_var_d_var(self,parentsvarparamstuple):
        sigma=np.sqrt(np.exp(parentsvarparamstuple[1]))
        mu=parentsvarparamstuple[0]
        d_C11_d_var=sigma**2
        d_C12_d_var=2*mu*sigma**2
        d_C22_d_var=4*(mu*sigma)**2+2*sigma**4
        return d_C11_d_var*(self.params[1]**2) + d_C22_d_var*(self.params[2]**2) + 2*d_C12_d_var*(self.params[1]*self.params[2])
'''