import torch.nn as nn
from .Multiscale2D_parts import Conv2DRelu
from .Multiscale2D_parts import EndecBlock2D
from .Multiscale2D_parts import Up2D
from .Multiscale2D_parts import Up2D_2
from .Multiscale2D_parts import Conv2DSigmoid
from .Multiscale2D_parts import Conv2DSoftmax
from .Multiscale2D_parts import ConvLstm2DRelu
from .Multiscale2D_parts import PSPModule

class MultiscaleNet2D(nn.Module):
    def __init__(self, n_channels, p_dropout=0.2, up_mode='base', activation='sigmoid', attention=True):
        super(MultiscaleNet2D, self).__init__()
        
        self.n_channels = n_channels
        self.p_dropout  = p_dropout
        self.attention   = attention
        
        #~~~~~~~~~~~~~~~~~~~ Encoder ~~~~~~~~~~~~~~~~~~~~~~
        
        self.conv2DRelu0 =     Conv2DRelu(self.n_channels, 16, stride=(1, 1))
        #Block 1
        self.endecBlock1 =   EndecBlock2D(16, 16)
        self.conv2DRelu1 =     Conv2DRelu(16, 32)
        #Block 2
        self.endecBlock2 =   EndecBlock2D(32, 16)
        self.conv2DRelu2 =     Conv2DRelu(16, 32)
        #Block 3
        self.endecBlock3 =   EndecBlock2D(32, 16)
        self.conv2DRelu3 =     Conv2DRelu(16, 32)
        #Block convLSTM
        self.convLSTM    = ConvLstm2DRelu(32, hidden_dim=[16, 16, 16]) #entran 32 canales y salen 16
        
        
        #~~~~~~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~
        
        if(up_mode == 'base'):
            self.upBlock1    =           Up2D(16, 64, kernel_size_convT=(3, 3), stride_convT=(1, 1), padding_convT=(1,1))
            self.upBlock2    =           Up2D(64, 64)
            self.upBlock3    =           Up2D(64, 64)
            self.upBlock4    =         Up2D_2(64, 16, p_dropout) 
        elif(up_mode == 'M2'):
            self.upBlock1    =           Up2D(16, 32, kernel_size_convT=(3, 3), stride_convT=(1, 1), padding_convT=(1,1), attention=self.attention)
            self.upBlock2    =           Up2D(32, 32, attention=self.attention)
            self.upBlock3    =           Up2D(32, 32, attention=self.attention)
            self.upBlock4    =         Up2D_2(32, 16, p_dropout, attention=self.attention)
        else:
            raise NotImplementedError('Unknown up_mode function.')


        if(  activation =='sigmoid'):
            self.activation  =  Conv2DSigmoid(16,  1, kernel_size=(3, 3), stride=(1, 1)) 
        elif(activation == 'softmax'):
            self.activation  =  Conv2DSoftmax(16,  2, kernel_size=(3, 3), stride=(1, 1))
        else:
            raise NotImplementedError('Unknown activation function.')
        
    #----------------------------------------------------------------------------------------
    # Forward
    #----------------------------------------------------------------------------------------
        
    def forward(self, x):
        
        #~~~~~~~~~~~~~~~~~~~ Encoder ~~~~~~~~~~~~~~~~~~~~~~
        
        x0      = self.conv2DRelu0(x)
        
        x1, x1_ = self.endecBlock1(x0)
        x1_     = self.conv2DRelu1(x1_)
        
        x2, x2_ = self.endecBlock2(x1_)
        x2_     = self.conv2DRelu2(x2_)
        
        x3, x3_ = self.endecBlock3(x2_)
        x3_     = self.conv2DRelu3(x3_)
        
        x       = self.convLSTM(x3_)

        #~~~~~~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~

        x       = self.upBlock1(x, x3_, x3)
        x       = self.upBlock2(x, x2_, x2)
        x       = self.upBlock3(x, x1_, x1)
        x       = self.upBlock4(x, x0)
        x       = self.activation(x)
        
        return x   