'''
Unit tests for inference (linear, variational = E-Step, Laplace, sampling, quadrature) if all maps are linear: in this case the results should be the same

Additional tests for E-Step

Can be called
'''
from unittest import  TestCase,main
import GaussianBeliefNetwork as GBN
import GaussianRandomVariable as GRV
import MeanMap as MM
import numpy as np
class test_inference(TestCase):
    def test_linear_inference_2var(self):
        # checks inference output, all meanmaps are linear
        # for two children
        vt=GRV.GaussianRandomVariable('vt',[],[],1)
        v1=GRV.GaussianRandomVariable('v1',[vt],[MM.LinearMeanMap([0,1])],1)
        v2=GRV.GaussianRandomVariable('v2',[vt],[MM.LinearMeanMap([0,1])],1)
        PGBN=GBN.PyramidGaussianBeliefNetwork(vt,[v1,v2])
        
        (mean,cov,names)= PGBN.inference({'vt':0})
        self.assertAlmostEqual(mean[names.index('v1')],0)
        self.assertAlmostEqual(mean[names.index('v2')],0)
        self.assertAlmostEqual(np.linalg.norm(cov-np.diag([1,1])),0)
        
        (mean,cov,names)= PGBN.inference({'vt':3})
        self.assertAlmostEqual(mean[names.index('v1')],3)
        self.assertAlmostEqual(mean[names.index('v2')],3)
        self.assertAlmostEqual(np.linalg.norm(cov-np.diag([1,1])),0)
        
        (mean,cov,names)= PGBN.inference({'v1':1})
        self.assertAlmostEqual(mean[names.index('vt')],0.5)
        self.assertAlmostEqual(mean[names.index('v2')],0.5)
        self.assertAlmostEqual(cov[names.index('vt'),names.index('vt')],0.5)
        self.assertAlmostEqual(cov[names.index('vt'),names.index('v2')],0.5)
        self.assertAlmostEqual(cov[names.index('v2'),names.index('v2')],1+0.5)
    def test_linear_inference_3var(self):
        # checks inference output, all meanmaps are linear
        # for three children
        vt=GRV.GaussianRandomVariable('vt',[],[],1)
        v1=GRV.GaussianRandomVariable('v1',[vt],[MM.LinearMeanMap([0,1])],1)
        v2=GRV.GaussianRandomVariable('v2',[vt],[MM.LinearMeanMap([1,2])],1)
        v3=GRV.GaussianRandomVariable('v3',[vt],[MM.LinearMeanMap([0,1])],2)
        PGBN=GBN.PyramidGaussianBeliefNetwork(vt,[v1,v2,v3])
        
        (mean,cov,names)= PGBN.inference({'vt':0})
        self.assertAlmostEqual(mean[names.index('v1')],0)
        self.assertAlmostEqual(mean[names.index('v2')],1)
        self.assertAlmostEqual(mean[names.index('v3')],0)
        self.assertAlmostEqual(np.linalg.norm(cov-np.diag([1,1,2])),0)
        
        (mean,cov,names)= PGBN.inference({'v1':0,'v2':1,'v3':0})
        self.assertAlmostEqual(mean[names.index('vt')],0)
    def test_estep_linear_3var(self):
        # checks E-step variational distribution, all meanmaps are linear
        # for three children
        vt=GRV.GaussianRandomVariable('vt',[],[],1)
        v1=GRV.GaussianRandomVariable('v1',[vt],[MM.LinearMeanMap([0,1])],1)
        v2=GRV.GaussianRandomVariable('v2',[vt],[MM.LinearMeanMap([1,2])],1)
        v3=GRV.GaussianRandomVariable('v3',[vt],[MM.LinearMeanMap([0,1])],2)
        PGBN=GBN.PyramidGaussianBeliefNetwork(vt,[v1,v2,v3])
        valuesdict={'v1':0,'v2':1,'v3':0}
        (mean,cov,names)= PGBN.inference(valuesdict)
        suffstat=PGBN.E_Step_one_instance(valuesdict)[1]
        self.assertAlmostEqual(suffstat['vt']['mu'],mean[names.index('vt')])
        self.assertAlmostEqual(suffstat['vt']['e'],cov[names.index('vt'),names.index('vt')])

        valuesdict={'v1':2,'v2':2,'v3':1}
        (mean,cov,names)= PGBN.inference(valuesdict)
        suffstat=PGBN.E_Step_one_instance(valuesdict)[1]
        self.assertAlmostEqual(suffstat['vt']['mu'],mean[names.index('vt')])
        self.assertAlmostEqual(suffstat['vt']['e'],cov[names.index('vt'),names.index('vt')])
        
        valuesdict={'v1':-3,'v2':0,'v3':-1}
        (mean,cov,names)= PGBN.inference(valuesdict)
        suffstat=PGBN.E_Step_one_instance(valuesdict)[1]
        self.assertAlmostEqual(suffstat['vt']['mu'],mean[names.index('vt')])
        self.assertAlmostEqual(suffstat['vt']['e'],cov[names.index('vt'),names.index('vt')])

        vt=GRV.GaussianRandomVariable('vt',[],[],1)
        v1=GRV.GaussianRandomVariable('v1',[vt],[MM.LinearMeanMap([-0.3,0.8])],0.05)
        v2=GRV.GaussianRandomVariable('v2',[vt],[MM.LinearMeanMap([1,2])],0.1)
        v3=GRV.GaussianRandomVariable('v3',[vt],[MM.LinearMeanMap([0,1.2])],0.8)
        PGBN=GBN.PyramidGaussianBeliefNetwork(vt,[v1,v2,v3])

        valuesdict={'v1':0.05,'v2':1.4,'v3':0.3}
        (mean,cov,names)= PGBN.inference(valuesdict)
        suffstat=PGBN.E_Step_one_instance(valuesdict)[1]
        self.assertAlmostEqual(suffstat['vt']['mu'],mean[names.index('vt')])
        self.assertAlmostEqual(suffstat['vt']['e'],cov[names.index('vt'),names.index('vt')])

        valuesdict={'v1':-2,'v2':-0.82,'v3':-1.3}
        (mean,cov,names)= PGBN.inference(valuesdict)
        suffstat=PGBN.E_Step_one_instance(valuesdict)[1]
        self.assertAlmostEqual(suffstat['vt']['mu'],mean[names.index('vt')])
        self.assertAlmostEqual(suffstat['vt']['e'],cov[names.index('vt'),names.index('vt')])

        valuesdict={'v1':2,'v2':2,'v3':1}
        (mean,cov,names)= PGBN.inference(valuesdict)
        suffstat=PGBN.E_Step_one_instance(valuesdict)[1]
        self.assertAlmostEqual(suffstat['vt']['mu'],mean[names.index('vt')])
        self.assertAlmostEqual(suffstat['vt']['e'],cov[names.index('vt'),names.index('vt')])
        
    def test_methods_3var(self):
        # check all methods when having three children, with either all or only two observed
        # sampling is only expected to converge!
        # the other ones are all exact (up to rounding/discretization)
        seed0=1
        np.random.seed(seed0)
        vt=GRV.GaussianRandomVariable('vt',[],[],1)
        v1=GRV.GaussianRandomVariable('v1',[vt],[MM.LinearMeanMap([0,1])],1)
        v2=GRV.GaussianRandomVariable('v2',[vt],[MM.LinearMeanMap([1,2])],1)
        v3=GRV.GaussianRandomVariable('v3',[vt],[MM.LinearMeanMap([0,1])],2)
        PGBN=GBN.PyramidGaussianBeliefNetwork(vt,[v1,v2,v3])
        valuesdicts=[{'v1':0,'v2':1,'v3':0},{'v1':0.1,'v2':1.9,'v3':-0.2},{'v1':0.4,'v2':2},{'v1':-0.8,'v3':-1.5}]
        intervalsizes=[False,0.1]
        for valuesdict in valuesdicts:
            for intervalsize in intervalsizes:
                intervli=PGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='linear')
                intervla=PGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='laplace')
                self.assertAlmostEqual(intervli[0],intervla[0])
                self.assertAlmostEqual(intervli[1],intervla[1])
                self.assertAlmostEqual(intervli[2],intervla[2])
                intervva=PGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='variational')
                self.assertAlmostEqual(intervli[0],intervva[0])
                self.assertAlmostEqual(intervli[1],intervva[1])
                self.assertAlmostEqual(intervli[2],intervva[2])
                intervqu=PGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='quadrature')
                self.assertAlmostEqual(intervli[0],intervqu[0])
                self.assertAlmostEqual(intervli[1],intervqu[1])
                self.assertAlmostEqual(intervli[2],intervqu[2])
                intervsa=PGBN.interval_conditional_density_topnode(valuesdict,intervalsize,method='sampling',samplesize=1e4)
                self.assertAlmostEqual(intervli[0],intervsa[0],places=1)#only approximate
                self.assertAlmostEqual(intervli[1],intervsa[1],places=1)
                self.assertAlmostEqual(intervli[2],intervsa[2],places=1)

if __name__=='__main__':
    main()