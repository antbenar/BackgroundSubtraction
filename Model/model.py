import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

#from parts import Attention
from parts import Conv3DRelu
from parts import ConvLstm2DRelu
from parts import EndecBlock
from parts import Up
from parts import Up2


class Net(nn.Module):
    def __init__(self, n_channels, p_dropout=0.2):
        super(Net, self).__init__()
        
        self.n_channels = n_channels
        self.p_dropout = p_dropout

        #~~~~~~~~~~~~~~~~~~~ Encoder ~~~~~~~~~~~~~~~~~~~~~~
        
        self.conv3DRelu0 = Conv3DRelu(self.n_channels, 16)
        #Block 1
        self.endecBlock1 = EndecBlock(16, 16)
        self.conv3DRelu1 = Conv3DRelu(16, 32)
        #Block 2
        self.endecBlock2 = EndecBlock(32, 16)
        self.conv3DRelu2 = Conv3DRelu(16, 32)
        #Block 3
        self.endecBlock3 = EndecBlock(32, 16)
        self.conv3DRelu3 = Conv3DRelu(16, 32)
        #Block convLSTM
        self.convLSTM    = ConvLstm2DRelu(32, hidden_dim=[8, 8, 16]) #entran 32 canales y salen 16
        
        #~~~~~~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~
        
        #BlockDec 1
        self.upBlock1    = Up(16, 64)
        #BlockDec 2
        self.upBlock2    = Up(64, 64)
        #BlockDec 3
        self.upBlock3    = Up(64, 64)
        #BlockDec 4
        self.upBlock4    = Up2(64, 32, p_dropout)
        
        self.activation  = Conv3DRelu(in_channels=32, out_channels=1, kernel_size=(2, 3, 3), stride=(1, 1, 1))
        
        
    #~~~~~~~~~~~~~~~~~~~ Forward ~~~~~~~~~~~~~~~~~~~~~~
    
    def forward(self, x):
        
        #~~~~~~~~~~~~~~~~~~~ Encoder ~~~~~~~~~~~~~~~~~~~~~~
        
        x0      = self.conv3DRelu0(x)
        
        x1, x1_ = self.endecBlock1(x0)
        x1_     = self.conv3DRelu1(x1_)
        
        x2, x2_ = self.endecBlock2(x1_)
        x2_     = self.conv3DRelu2(x2_)
        
        x3, x3_ = self.endecBlock3(x2_)
        x3_     = self.conv3DRelu3(x3_)
        
        x       = self.convLSTM(x3_)
        
        #~~~~~~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~
        
        x       = self.upBlock1(x, x3_, x3)
        x       = self.upBlock2(x, x2_, x2)
        x       = self.upBlock3(x, x1_, x1)
        x       = self.upBlock4(x, x0)
        
        x       = self.activation(x)
        
        return x

# model static attributes
n_channels = 3
p_dropout=0.2

# instance the model
net = Net(
    n_channels, 
    p_dropout
)
print(net)



