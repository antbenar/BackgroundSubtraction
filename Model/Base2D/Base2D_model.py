import torch.nn as nn
from .Base2D_parts import Conv2DRelu
from .Base2D_parts import EndecBlock2D
from .Base2D_parts import Up2D
from .Base2D_parts import Up2D_2
from .Base2D_parts import Conv2DSigmoid

class Net(nn.Module):
    def __init__(self, n_channels, p_dropout=0.2):
        super(Net, self).__init__()
        
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
        #Block 4
        self.conv2DRelu4 =     Conv2DRelu(32, 64, stride=(1, 1))
        
        #~~~~~~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~
        
        #BlockDec 1
        self.upBlock1    =           Up2D(64, 64, kernel_size_convT=(3, 3), stride_convT=(1, 1), padding_convT=(1,1))
        #BlockDec 2
        self.upBlock2    =           Up2D(64, 64)
        #BlockDec 3
        self.upBlock3    =           Up2D(64, 64)
        #BlockDec 4 
        self.upBlock4    =         Up2D_2(64, 16, p_dropout)
        
        self.activation  =  Conv2DSigmoid(16,  1, kernel_size=(3, 3), stride=(1, 1))
        
        
        
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
        
        x       = self.conv2DRelu4(x3_)
        
        #~~~~~~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~

        x       = self.upBlock1(x, x3_, x3)
        x       = self.upBlock2(x, x2_, x2)
        x       = self.upBlock3(x, x1_, x1)
        x       = self.upBlock4(x, x0)
        x       = self.activation(x)
        
        return x   

    #----------------------------------------------------------------------------------------
    # Forward with print out channels
    #----------------------------------------------------------------------------------------
        
    # def forward(self, x):
        
    #     #~~~~~~~~~~~~~~~~~~~ Encoder ~~~~~~~~~~~~~~~~~~~~~~
        
    #     print("tensor - input \t\t= ",x.size())
    #     x0      = self.conv2DRelu0(x)
    #     print("tensor - x0 \t\t= ",x0.size())
        
    #     x1, x1_ = self.endecBlock1(x0)
    #     x1_     = self.conv2DRelu1(x1_)
    #     print("tensor - x1_ \t\t= ",x1_.size())
        
    #     x2, x2_ = self.endecBlock2(x1_)
    #     x2_     = self.conv2DRelu2(x2_)
    #     print("tensor - x2_ \t\t= ",x2_.size())
        
    #     x3, x3_ = self.endecBlock3(x2_)
    #     x3_     = self.conv2DRelu3(x3_)
    #     print("tensor - x3_ \t\t= ",x3_.size())
        
    #     x       = self.conv2DRelu4(x3_)
    #     print("tensor - x4_ \t\t= ",x.size())
        
    #     #~~~~~~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~

    #     x       = self.upBlock1(x, x3_, x3)
    #     print("tensor - upBlock1 \t= ",x.size())
        
    #     x       = self.upBlock2(x, x2_, x2)
    #     print("tensor - upBlock2 \t= ",x.size())
        
    #     x       = self.upBlock3(x, x1_, x1)
    #     print("tensor - upBlock3 \t= ",x.size())
        
    #     x       = self.upBlock4(x, x0)
    #     print("tensor - upBlock4 \t= ",x.size())
        
    #     x       = self.activation(x)
    #     print("tensor - output   \t= ",x.size())
        
    #     return x    
