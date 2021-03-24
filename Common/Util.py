import os
import glob
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy             as np
from random              import shuffle

""" Averager class
    --------------
    Average of array or scalar
"""
class Averager():
    def __init__(self,n=0):
        if n >0:
            self.mean = np.array(n)
        else:    
            self.mean  = 0
        self.count = 0
        self.n = n
    def reset(self):
        if self.n >0:
            self.mean = np.array(self.n)
        else:    
            self.mean  = 0
        self.count = 0
    def update(self,val):
        n = self.count
        self.count = n + 1
        self.mean  = (self.mean*n + val)/self.count
    def val(self):
        return self.mean