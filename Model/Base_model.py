import torch.nn as nn
#from Base_parts      import Attention
from Model.Base_parts import Conv3DRelu
from Model.Base_parts import ConvLstm2DRelu
from Model.Base_parts import EndecBlock
from Model.Base_parts import UpM1
from Model.Base_parts import UpM2
from Model.Base_parts import UpM3
from Model.Base_parts import Up2
from Model.Base_parts import Conv3DSigmoid

class Net(nn.Module):
    def __init__(self, n_channels, p_dropout=0.2, up_mode='base'):
        super(Net, self).__init__()
        
        self.n_channels = n_channels
        self.p_dropout = p_dropout

        #~~~~~~~~~~~~~~~~~~~ Encoder ~~~~~~~~~~~~~~~~~~~~~~
        
        self.conv3DRelu0 =     Conv3DRelu(self.n_channels, 16, stride=(1, 1, 1))
        #Block 1
        self.endecBlock1 =     EndecBlock(16, 16)
        self.conv3DRelu1 =     Conv3DRelu(16, 32)
        #Block 2
        self.endecBlock2 =     EndecBlock(32, 16)
        self.conv3DRelu2 =     Conv3DRelu(16, 32)
        #Block 3
        self.endecBlock3 =     EndecBlock(32, 16)
        self.conv3DRelu3 =     Conv3DRelu(16, 32)
        #Block convLSTM
        self.convLSTM    = ConvLstm2DRelu(32, hidden_dim=[16, 16, 16]) #entran 32 canales y salen 16
        
        #~~~~~~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~
        
        if(up_mode == 'base'):
            self.upBlock1    =         UpM1(16, 64, kernel_size_convT=(3, 3, 3), stride_convT=(1, 1, 1), padding_convT=(1,1,1))
            self.upBlock2    =         UpM1(64, 64)
            self.upBlock3    =         UpM1(64, 64)
            self.upBlock4    =          Up2(64, 32, p_dropout)    
        elif(up_mode == 'M2'):
            self.upBlock1    =         UpM2(16, 32, kernel_size_convT=(3, 3, 3), stride_convT=(1, 1, 1), padding_convT=(1,1,1))
            self.upBlock2    =         UpM2(32, 32)
            self.upBlock3    =         UpM2(32, 32)
            self.upBlock4    =          Up2(32, 32, p_dropout)    
        elif(up_mode == 'M3'):
            self.upBlock1    =         UpM3(16, 64, kernel_size_convT=(3, 3, 3), stride_convT=(1, 1, 1), padding_convT=(1,1,1))
            self.upBlock2    =         UpM3(64, 64)
            self.upBlock3    =         UpM3(64, 64)
            self.upBlock4    =          Up2(64, 32, p_dropout)    
        else:
            raise NotImplementedError('Unknown operation function.')
        
        
        self.activation  =  Conv3DSigmoid(32,  1, kernel_size=(1, 3, 3), stride=(1, 1, 1))
        
        
    #----------------------------------------------------------------------------------------
    # Forward
    #----------------------------------------------------------------------------------------
        
    def forward(self, x):
        
        #~~~~~~~~~~~~~~~~~~~ Encoder ~~~~~~~~~~~~~~~~~~~~~~
        
        x0      = self.conv3DRelu0(x)
        #Block 1
        x1, x1_ = self.endecBlock1(x0)
        x1_     = self.conv3DRelu1(x1_)
        #Block 2
        x2, x2_ = self.endecBlock2(x1_)
        x2_     = self.conv3DRelu2(x2_)
        #Block 3
        x3, x3_ = self.endecBlock3(x2_)
        x3_     = self.conv3DRelu3(x3_)
        #Block convLSTM
        x       = self.convLSTM(x3_)
        
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
        
    """
    def forward(self, x):
        
        #~~~~~~~~~~~~~~~~~~~ Encoder ~~~~~~~~~~~~~~~~~~~~~~
        
        #print("tensor - input \t\t= ",x.size())
        x0      = self.conv3DRelu0(x)
        #print("tensor - x0 \t\t= ",x0.size())
        
        x1, x1_ = self.endecBlock1(x0)
        x1_     = self.conv3DRelu1(x1_)
        #print("tensor - x1_ \t\t= ",x1_.size())
        
        x2, x2_ = self.endecBlock2(x1_)
        x2_     = self.conv3DRelu2(x2_)
        #print("tensor - x2_ \t\t= ",x2_.size())
        
        x3, x3_ = self.endecBlock3(x2_)
        x3_     = self.conv3DRelu3(x3_)
        #print("tensor - x3_ \t\t= ",x3_.size())
        
        x       = self.convLSTM(x3_)
        #print("tensor - convLSTM \t= ",x.size())
        
        #~~~~~~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~

        x       = self.upBlock1(x, x3_, x3)
        #print("tensor - upBlock1 \t= ",x.size())
        
        x       = self.upBlock2(x, x2_, x2)
        #print("tensor - upBlock2 \t= ",x.size())
        
        x       = self.upBlock3(x, x1_, x1)
        #print("tensor - upBlock3 \t= ",x.size())
        
        x       = self.upBlock4(x, x0)
        #print("tensor - upBlock4 \t= ",x.size())
        
        x       = self.activation(x)
        #print("tensor - output   \t= ",x.size())
        
        return x
    """

