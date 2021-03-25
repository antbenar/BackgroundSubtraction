import torch
import torch.nn     as nn
from Model.ConvLstm import ConvLSTM

#----------------------------------------------------------------------------------------
# Conv3DSigmoid
#----------------------------------------------------------------------------------------
      
class Conv3DSigmoid(nn.Module):
    """Convolution 3D with activation relu"""

    def __init__(self, in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        padding      = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)
        self.conv3d  = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_tensor) :
        x = self.conv3d(input_tensor)
        x = self.sigmoid(x)
        return x
    
#----------------------------------------------------------------------------------------
# Conv3DRelu
#----------------------------------------------------------------------------------------
      
class Conv3DRelu(nn.Module):
    """Convolution 3D with activation relu"""

    def __init__(self, in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2)):
        super().__init__()
        padding     = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu   = nn.ReLU()
        
    def forward(self, input_tensor) :
        x = self.conv3d(input_tensor)
        x = self.relu(x)
        return x
    

#----------------------------------------------------------------------------------------
# ConvLstm2DRelu
#----------------------------------------------------------------------------------------
      
class ConvLstm2DRelu(nn.Module):
    """Convolution 3D with activation relu"""

    def __init__(self, in_channels, hidden_dim=[16, 16, 16], kernel_size=(3, 3), num_layers=3, batch_first=True, bias=True, return_all_layers=True):
        super().__init__()
        self.convLstm2D = ConvLSTM(input_dim=in_channels, 
                                   hidden_dim=hidden_dim, 
                                   kernel_size=kernel_size,
                                   num_layers=num_layers,
                                   batch_first=batch_first,
                                   bias=bias, 
                                   return_all_layers=return_all_layers
                          )
        self.relu       = nn.ReLU()
        
    def forward(self, input_tensor) :
        #permute in form (b, c, t, h, w) -> (b, t, c, h, w)
        input_tensor = input_tensor.permute(0, 2, 1, 3, 4)
        
        x, _ = self.convLstm2D(input_tensor)
        
        #permute back (b, t, c, h, w) -> (b, c, t, h, w)
        x       = x[0].permute(0, 2, 1, 3, 4)
        
        x       = self.relu(x)
        return x
    
    
#----------------------------------------------------------------------------------------
# EndecBlock 
#----------------------------------------------------------------------------------------
      
class EndecBlock(nn.Module):
    """Blocks of encoder (Mini encoder-decoders)"""

    def __init__(self, in_channels=16, out_channels=16):
        super().__init__()
        self.channels_concat    = in_channels + 16
        
        self.conv3DRelu1        = Conv3DRelu(in_channels, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.convTranspose3d    = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1,1,1))
        #self.convTranspose3d    = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1,2,1))

        self.batchNormalization = nn.BatchNorm3d(num_features=self.channels_concat)
        self.conv3DRelu2        = Conv3DRelu(self.channels_concat, out_channels=out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1))

    def forward(self, input_tensor) :
        x  = self.conv3DRelu1(input_tensor)
        x_ = self.convTranspose3d(x)
        x_ = torch.cat([input_tensor, x_], dim=1)
        x_ = self.batchNormalization(x_)
        x_ = self.conv3DRelu2(x_)
        return x, x_
    
    
#----------------------------------------------------------------------------------------
# Attention 
#----------------------------------------------------------------------------------------
        
class Attention(nn.Module):
    """Attention module"""

    def __init__(self, in_channels_tensor, in_channels_att_tensor, out_channels):
        super().__init__()
        self.conv3d1     = nn.Conv3d(in_channels_tensor, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d2     = nn.Conv3d(in_channels_att_tensor, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d3     = nn.Conv3d(in_channels_att_tensor, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.relu       = nn.ReLU()
        self.sigmoid    = nn.Sigmoid()
        
    def forward(self, tensor, att_tensor):
        g1 = self.conv3d1(tensor)
        x1 = self.conv3d2(att_tensor)
        x = g1.add(x1)
        x = self.relu(x)
        x = self.conv3d3(x)
        x = self.sigmoid(x)
        #print("tensor - x = ",x.size(), ", att_tensor = ",att_tensor.size())
        x = x * att_tensor
        #print("tensor - x = ",x.size())
        return x
    
#----------------------------------------------------------------------------------------
# PARTS OF THE UPSAMPLING OF THE MODEL 1
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
# Up 
#----------------------------------------------------------------------------------------

class UpM1(nn.Module):
    """Blocks of decoder"""

    def __init__(self, in_channels=64,  out_channels=64, kernel_size_convT=(3, 2, 2), stride_convT=(1, 2, 2), padding_convT=(1,0,0)):
        super().__init__()
        #To obtain the number of output chanels, the last convolution must be equal to the 
        #difference of channels of the result with the concatenation tensor
        out_channels -= 32
        #                                                                                                                      
        self.convTranspose3d    = nn.ConvTranspose3d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size_convT, stride=stride_convT, padding=padding_convT)
        #self.convTranspose3d    = nn.ConvTranspose3d(in_channels=in_channels, out_channels=16, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1, 1, 0))
        self.atention           = Attention(in_channels_tensor=32, in_channels_att_tensor=16, out_channels=16)
        #concat ->      32 + 16 = 48 channels
        self.conv3D             = nn.Conv3d(in_channels=48, out_channels=out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.batchNormalization = nn.BatchNorm3d(out_channels)
        self.relu               = nn.ReLU()
        #concat ->      32 + 32 = 64 channels
        
    def forward(self, input_tensor, prev_tensor1, prev_tensor2) :
        x = self.convTranspose3d(input_tensor)        
        x = self.atention(prev_tensor1, x)
        x = torch.cat([prev_tensor1, x], dim=1)
        x = self.conv3D(x)
        x = self.batchNormalization(x)
        x = self.relu(x)
        x = torch.cat([x, prev_tensor2], dim=1)
        return x
    
    
#----------------------------------------------------------------------------------------
# Up2
#----------------------------------------------------------------------------------------

class Up2(nn.Module):
    """Blocks of decoder"""

    def __init__(self, in_channels=64,  out_channels=32, p_dropout=0.2):
        super().__init__()
        self.convTranspose3d    = nn.ConvTranspose3d(in_channels=in_channels, out_channels=16, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        self.atention           = Attention(in_channels_tensor=16, in_channels_att_tensor=16, out_channels=16)
        #concat ->      16 + 16 = 32 channels
        self.conv3D             = nn.Conv3d(in_channels=32, out_channels=out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.batchNormalization = nn.BatchNorm3d(out_channels)
        self.relu               = nn.ReLU()
        self.dropout            = nn.Dropout(p=p_dropout)
        
    def forward(self, input_tensor, prev_tensor) :
        x = self.convTranspose3d(input_tensor)
        x = self.atention(prev_tensor, x)
        x = torch.cat([prev_tensor, x], dim=1)
        x = self.conv3D(x)
        x = self.batchNormalization(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
    
    
#----------------------------------------------------------------------------------------
# PARTS OF THE UPSAMPLING OF THE MODEL 2
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
# Up 
# --- PARTS OF THE UPSAMPLING OF THE MODEL 2
#----------------------------------------------------------------------------------------    
    
class UpM2(nn.Module):
    """Blocks of decoder"""

    def __init__(self, in_channels=32,  out_channels=32, kernel_size_convT=(3, 2, 2), stride_convT=(1, 2, 2), padding_convT=(1,0,0)):
        super().__init__()
        self.convTranspose3d    = nn.ConvTranspose3d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size_convT, stride=stride_convT, padding=padding_convT)
        #self.convTranspose3d    = nn.ConvTranspose3d(in_channels=in_channels, out_channels=16, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1, 1, 0))
        self.atention           = Attention(in_channels_tensor=32, in_channels_att_tensor=16, out_channels=16)
        #concat ->      32 + 16 = 48 channels
        self.conv3D             = nn.Conv3d(in_channels=48, out_channels=out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.batchNormalization = nn.BatchNorm3d(out_channels)
        self.relu               = nn.ReLU()
        
    def forward(self, input_tensor, prev_tensor1, prev_tensor2) :
        x = self.convTranspose3d(input_tensor)        
        x = self.atention(prev_tensor1, x)
        x = torch.cat([prev_tensor1, x], dim=1)
        x = self.conv3D(x)
        x = self.batchNormalization(x)
        x = self.relu(x)
        return x
    
#----------------------------------------------------------------------------------------
# Up 
# --- PARTS OF THE UPSAMPLING OF THE MODEL 3
#----------------------------------------------------------------------------------------

class UpM3(nn.Module):
    """Blocks of decoder"""

    def __init__(self, in_channels=32,  out_channels=32, kernel_size_convT=(3, 2, 2), stride_convT=(1, 2, 2), padding_convT=(1,0,0)):
        super().__init__()
        self.convTranspose3d    = nn.ConvTranspose3d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size_convT, stride=stride_convT, padding=padding_convT)
        #self.convTranspose3d    = nn.ConvTranspose3d(in_channels=in_channels, out_channels=16, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1, 1, 0))
        self.atention           = Attention(in_channels_tensor=32, in_channels_att_tensor=16, out_channels=16)
        #concat -> 32 + 32 + 16 = 80 channels
        self.conv3D             = nn.Conv3d(in_channels=80, out_channels=out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.batchNormalization = nn.BatchNorm3d(out_channels)
        self.relu               = nn.ReLU()
        
    def forward(self, input_tensor, prev_tensor1, prev_tensor2) :
        x = self.convTranspose3d(input_tensor)        
        x = self.atention(prev_tensor1, x)
        x = torch.cat([prev_tensor2, prev_tensor1, x], dim=1)
        x = self.conv3D(x)
        x = self.batchNormalization(x)
        x = self.relu(x)
        return x
    
    
    
    
    
