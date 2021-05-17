import torch
import torch.nn              as nn
import torch.nn.functional   as F
from Model.ConvLstm          import ConvLSTM



#----------------------------------------------------------------------------------------
# conv1x1
#----------------------------------------------------------------------------------------
 
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

