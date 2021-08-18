import torch.nn as nn
import torch
from ..Base2D.Base2D_parts import Conv2DRelu
from ..Base2D.Base2D_parts import EndecBlock2D
from ..Base2D.Base2D_parts import Up2D
from ..Base2D.Base2D_parts import Up2D_2
from ..Base2D.Base2D_parts import Conv2DSigmoid
from ..Base2D.Base2D_parts import Conv2DSoftmax
from ..Base2D.Base2D_parts import ConvLstm2DRelu


class Net2D(nn.Module):
    def __init__(self, n_channels, p_dropout=0.2, up_mode='base', activation='sigmoid'):
        super(Net2D, self).__init__()
        
        self.n_channels = n_channels
        self.p_dropout = p_dropout

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
            self.upBlock1    =           Up2D(16, 64, kernel_size_convT=(3, 3), stride_convT=(1, 1), padding_convT=(1,1))
            self.upBlock2    =           Up2D(64, 64)
            self.upBlock3    =           Up2D(64, 64)
            self.upBlock4    =         Up2D_2(64, 16, p_dropout)
        else:
            raise NotImplementedError('Unknown up_mode function.')

        
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
        
        # x3, x3_ = self.endecBlock3(x2_)
        # x3_     = self.conv2DRelu3(x3_)
        # x       = self.convLSTM(x3_)
        x       = self.convLSTM(x2_)

        #~~~~~~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~

        # x       = self.upBlock1(x, x3_, x3)
        # x       = self.upBlock2(x, x2_, x2)
        # x       = self.upBlock3(x, x1_, x1)
        # x       = self.upBlock4(x, x0)
        # x       = self.activation(x)
        
        x       = self.upBlock1(x, x2_, x2)
        x       = self.upBlock2(x, x1_, x1)
        x       = self.upBlock4(x, x0)        
        
        return x   


class LoGoNet2D(nn.Module):
    def __init__(self, n_channels, p_dropout=0.2, up_mode='base', activation='sigmoid'):
        super(LoGoNet2D, self).__init__()
        
        self.n_channels = n_channels
        self.p_dropout = p_dropout

        #~~~~~~~~~~~~~~~~~~~ Net ~~~~~~~~~~~~~~~~~~~~~~
        
        self.baseNet = Net2D(
                self.n_channels, 
                self.p_dropout,
                up_mode    = up_mode,
                activation = activation
            )
        
        #~~~~~~~~~~~~~~~~~~~ Convs ~~~~~~~~~~~~~~~~~~~~~~
        
        self.conv1 = nn.Conv2d(n_channels, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(8, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(8)
        
        self.relu = nn.ReLU(inplace=True)
        
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
        
        xin = x.clone()
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        
        x = self.baseNet(x)
        
        x_loc = x.clone()
        b, c, h, w = xin.size()
        h, w       = int(h/4), int(w/4)

        for i in range(0,4):
            for j in range(0,4):

                x_p = xin[:,:,h*i:h*(i+1),w*j:w*(j+1)]
                
                x_p = self.baseNet(x_p)
                
                x_loc[:,:,h*i:h*(i+1),w*j:w*(j+1)] = x_p

        x = torch.add(x,x_loc)
        x = self.activation(x)

        return x   
