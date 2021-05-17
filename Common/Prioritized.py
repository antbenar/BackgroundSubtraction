import pickle
import numpy as np
from   Common.SumTree import SumTree

#----------------------------------------------------------------------------------------
# PrioritizedSamples 
# from https://github.com/Suryavf/SelfDrivingCar/blob/master/common/prioritized.py
#----------------------------------------------------------------------------------------

class PrioritizedSamples(object):
    """ Constructor """
    def __init__(self,n_samples,alpha=1.0,beta=0.9,
                        betaLinear=False,betaPhase=50,
                        balance=False,c=1.0,
                        fill=False):
        # Parameters
        self.n_samples = n_samples
        self.alpha     =     alpha
        self.beta      =      beta

        self.n_leaf    = int(2**np.ceil(np.log2(n_samples)))
        self.n_nodes   = 2*self.n_leaf - 1

        if fill: _fill =  1.0
        else   : _fill = None

        # Samples Tree
        self.priority = SumTree( self.n_nodes,val=_fill,limit=n_samples )
        
        # Beta
        self.betaLinear = betaLinear
        if betaLinear: 
            self.beta_m  = (1.0-self.beta)/betaPhase
            self.beta_b  = self.beta
            self.n_iter  = 0

        # Upper Confidence Bound applied to trees (UCT)
        self.balance = balance
        if balance: 
            self.c = c
            self.sampleCounter = np.zeros( self.n_samples )
            self. totalCounter = 0
            self.UTC = SumTree( self.n_nodes,val=_fill,limit=n_samples )

    """ Save """
    def save(self,path='priority.pck'):
        p = dict()
        p['priority'] = self.priority
        if self.balance: 
            p['sampleCounter'] = self.sampleCounter
            p[ 'totalCounter'] = self.totalCounter
            p[          'UTC'] = self.UTC
        # Save
        with open(path, 'wb') as handle:
            pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """ Load """
    def load(self,path='priority.pck'):
        with open(path, 'rb') as handle:
            p = pickle.load(handle)
        self.priority = p['priority']
        if self.balance: 
            self.sampleCounter = p['sampleCounter']
            self.totalCounter  = p[ 'totalCounter']
            self.UTC           = p          ['UTC']

    """ Step """
    def step(self):
        if self.betaLinear: 
            self.beta = self.beta_b + self.beta_m*self.n_iter
            self.beta = min(self.beta,1.0)
            self.n_iter += 1

        # Update UCT
        if self.balance:
            for idx in range(self.n_samples):
                p = self.priority[idx]/self.priority.sum()
                u = self.c*np.sqrt( np.log(self.totalCounter) / ( 1 + self.sampleCounter[idx]) )
                self.UTC[idx] = p+u
                

    """ Functions """
    def update(self,idx,p = None):
        # Prioritized sampling
        self.priority[idx] = p**self.alpha

        # UCT
        if self.balance:
            self.sampleCounter[idx]+=1
            self. totalCounter     +=1

            p = self.priority[idx]/self.priority.sum()
            u = self.c*np.sqrt( np.log(self.totalCounter) / ( 1 + self.sampleCounter[idx]) )

            self.UTC[idx] = p+u

    
    """ Get sample """
    def sample(self):
        if self.balance: tree = self.UTC
        else           : tree = self.priority
        # Roulette
        sp = np.random.uniform()
        sp = sp * tree.sum()

        # Index in pow(priority,alpha)
        idx = tree.search(sp)
        idx = idx - (self.n_leaf - 1)
        
        if self.beta > 0:
            # Probability
            prob = tree[idx]/tree.sum()
            # Importance-sampling (IS) weights
            weight = ( 1/(self.n_samples*prob) )**self.beta
        else:
            weight = 1 
        
        return int(idx),weight
        
