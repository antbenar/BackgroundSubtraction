import os
import glob
import argparse
import datetime
import torch 
import torch.nn          as nn
import matplotlib.pyplot as plt
import numpy             as np
from random              import shuffle
from torch.autograd      import Variable


#----------------------------------------------------------------------------------------
# Averager class
# --- Average of array or scalar
#----------------------------------------------------------------------------------------
      
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
    
    
#----------------------------------------------------------------------------------------
# Calculate Model Size
# --- Calculate inputs, model parameters and intermediate variables
#----------------------------------------------------------------------------------------
    
class ModelSize():
    def __init__(self, input_, model, device):
        self.input = input_
        self.model = model
        self.device = device
        self.bytes  = 4
        
        self.input_bytes    = self.calculateInput()
        self.parameter_bytes = self.calculateParameters()
        self.interm_variables_bits = self.calculateIntVariables()
        
        print('input:', self.input_bytes , 'Mb')
        print('parameter:', self.parameter_bytes , 'Mb')
        print('interm_variables:', self.interm_variables_bits, 'Mb')
        
        inp_param = (self.input_bytes + self.parameter_bytes)
        total = (inp_param  + self.interm_variables_bits)/1000
        print('Total + int:',total , 'Gb')
        
    def calculateInput(self): 
        inputs = torch.tensor(self.input.size())
        return torch.prod(inputs).item() * self.bytes //1000000
       
    
    def calculateParameters(self):        
        mods = list(self.model.modules())
        sizes = []
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        return self.sumBytes(sizes)//1000000
    
    def sumBytes(self, sizes):
        total_bytes = 0
        for i in range(len(sizes)):
            s = sizes[i]
            parameter_bytes = np.prod(np.array(s)) * self.bytes
            total_bytes += parameter_bytes
        
        return total_bytes
    
    def remove_sequential(self, model):
        for layer in model.modules():
            if type(layer) == nn.Sequential: # if sequential layer, apply recursively to layers in sequential layer
                self.remove_sequential(layer)
            if list(layer.children()) == []: # if leaf node, add it to list
                self.all_layers.append(layer)
            
    def calculateIntVariables(self):  
        input_ = self.input
        out_sizes = []
        self.all_layers = []
        self.remove_sequential(self.model)        
        conv2DVal = False 
        
        for i in range(0, len(self.all_layers)):
            m = self.all_layers[i]
            #print("="*40,m)
            #print(input_.size())

            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                input_size = torch.tensor(input_.size())
                input_size[1] = m.in_channels
                input_size = torch.Size(input_size.tolist())            
                input_ = Variable(torch.FloatTensor(input_size))
                
            elif isinstance(m, nn.BatchNorm3d):
                input_size = torch.tensor(input_.size())
                input_size[1] = m.num_features
                input_size = torch.Size(input_size.tolist())            
                input_ = Variable(torch.FloatTensor(input_size))
                
            elif isinstance(m, nn.Conv2d):
                input_size = torch.tensor(torch.Size([5, 16, 30, 40]))
                input_size[1] = m.in_channels
                input_size = torch.Size(input_size.tolist())     
                input_ = Variable(torch.FloatTensor(input_size))
                conv2DVal = True
            elif isinstance(m, nn.ReLU) and conv2DVal:
                input_size = torch.tensor(torch.Size([5, 16, 5, 30, 40]))
                input_size = torch.Size(input_size.tolist())     
                input_ = Variable(torch.FloatTensor(input_size))
                conv2DVal = True
                
            #print(input_.size())    
            out = m(input_)
                
            
            out_sizes.append(np.array(out.size()))
            input_ = out
            
        total_bytes   = self.sumBytes(out_sizes)
        total_bytes //=1000000
        total_bytes  *=2   # *2 -> we need to store values AND gradients
        
        return total_bytes
    
    
    
    
    