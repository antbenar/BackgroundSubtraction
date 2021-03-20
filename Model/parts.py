import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from ConvLstm import *

#~~~~~~~~~~~~~~~~~~~ Attention ~~~~~~~~~~~~~~~~~~~~~~

class Attention(nn.Module):
    """Attention module"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv3d     = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.relu       = nn.ReLU()
        self.sigmoid    = nn.Sigmoid()

    def add(self, tensor_a, tensor_b):
        return tf.add(tensor_a, tensor_b)

    def forward(self, tensor, att_tensor):
        g1 = self.conv3d(tensor)
        x1 = self.conv3d(att_tensor)
        
        x = self.add(g1, x1)
        x = self.relu(x)
        x = self.con3d(x)
        x = self.sigmoid(x)
        x = x * att_tensor

        return x



#~~~~~~~~~~~~~~~~~~~ Conv3DRelu ~~~~~~~~~~~~~~~~~~~~~~

class Conv3DRelu(nn.Module):
    """Convolution 3D with activation relu"""

    def __init__(self, in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        padding = (kernel_size[0]//2, kernel_size[0]//2, 0)
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu   = nn.ReLU()
        
    def forward(self, input_tensor) :
        return self.relu(self.conv3d(input_tensor))
    




#~~~~~~~~~~~~~~~~~~~ ConvLstm2DRelu ~~~~~~~~~~~~~~~~~~~~~~

class ConvLstm2DRelu(nn.Module):
    """Convolution 3D with activation relu"""

    def __init__(self, in_channels, hidden_dim=[8, 8, 16], kernel_size=(3, 3), num_layers=3, batch_first=True, bias=True, return_all_layers=True):
        super().__init__()
        self.convLstm2D = ConvLSTM(input_dim=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size,
                                   num_layers=num_layers, batch_first=batch_first, bias=bias, 
                                   return_all_layers=return_all_layers)
        self.relu       = nn.ReLU()
        
    def forward(self, input_tensor) :
        return self.relu(self.convLstm2D(input_tensor))
    
    
    
#~~~~~~~~~~~~~~~~~~~ EndecBlock ~~~~~~~~~~~~~~~~~~~~~~

class EndecBlock(nn.Module):
    """Blocks of encoder (Mini encoder-decoders)"""

    def __init__(self, in_channels=16, out_channels=16):
        super().__init__()
        self.channels_concat    = in_channels + 16
        
        self.conv3DRelu1        = Conv3DRelu(in_channels, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.convTranspose3d    = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1, 1, 0))
        self.batchNormalization = nn.BatchNorm2d(self.channels_concat)
        self.conv3DRelu2        = Conv3DRelu(self.channels_concat, out_channels=out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1))

    def forward(self, input_tensor) :
        x  = self.conv3DRelu1(input_tensor)
        x_ = self.convTranspose3d(x)
        x_ = torch.cat([input_tensor, x_], dim=-1)
        x_ = self.batchNormalization(x_)
        x_ = self.conv3DRelu2(x_)
        return x, x_
    
#~~~~~~~~~~~~~~~~~~~ Up ~~~~~~~~~~~~~~~~~~~~~~

class Up(nn.Module):
    """Blocks of decoder"""

    def __init__(self, in_channels=64,  out_channels=64):
        super().__init__()
        #To obtain the number of output chanels, the last convolution must be equal to the 
        #difference of channels of the result with the concatenation tensor
        out_channels -= 32
        
        self.convTranspose3d    = nn.ConvTranspose3d(in_channels=in_channels, out_channels=16, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1, 1, 0))
        self.atention           = Attention(in_channels=16, out_channels=16)
        #concat ->      32 + 16 = 48 channels
        self.conv3D             = nn.Conv3d(in_channels=48, out_channels=out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 0))
        self.batchNormalization = nn.BatchNorm2d(out_channels)
        self.relu               = nn.ReLU()
        #concat ->      32 + 32 = 64 channels
        
    def forward(self, input_tensor, prev_tensor1, prev_tensor2) :
        x = self.convTranspose3d(input_tensor)
        x = self.atention(x, prev_tensor1)
        x = torch.cat([prev_tensor1, x], dim=-1)
        x = self.conv3D(x)
        x = self.batchNormalization(x)
        x = self.relu(x)
        x = torch.cat([x, prev_tensor2], dim=-1)
        return x

#~~~~~~~~~~~~~~~~~~~ Up2 ~~~~~~~~~~~~~~~~~~~~~~

class Up2(nn.Module):
    """Blocks of decoder"""

    def __init__(self, in_channels=64,  out_channels=32, p_dropout=0.2):
        super().__init__()
        self.convTranspose3d    = nn.ConvTranspose3d(in_channels=in_channels, out_channels=16, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1, 1, 0))
        self.atention           = Attention(in_channels=16, out_channels=16)
        #concat ->      16 + 16 = 32 channels
        self.conv3D             = nn.Conv3d(in_channels=32, out_channels=out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 0))
        self.batchNormalization = nn.BatchNorm2d(out_channels)
        self.relu               = nn.ReLU()
        self.dropout            = nn.Dropout(p=p_dropout)
        
    def forward(self, input_tensor, prev_tensor) :
        x = self.convTranspose3d(input_tensor)
        x = self.atention(x, prev_tensor)
        x = torch.cat([prev_tensor, x], dim=-1)
        x = self.conv3D(x)
        x = self.batchNormalization(x)
        x = self.relu(x)
        x = self.dropout()
        return x