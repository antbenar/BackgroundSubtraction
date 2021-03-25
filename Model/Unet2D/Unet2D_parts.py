import torch
import torch.nn         as nn
import torch.functional as F
from Model.ConvLstm import ConvLSTM

#----------------------------------------------------------------------------------------
# UnetConv2
#----------------------------------------------------------------------------------------
      
class UnetConv2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1):
        super(UnetConv2, self).__init__()
        
        self.conv1 = nn.Sequential(
                                       nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(
                                       nn.Conv2d(out_size, out_size, kernel_size, stride, padding),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True)
                                   )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
    
    
#----------------------------------------------------------------------------------------
# AttentionBlock
#----------------------------------------------------------------------------------------

class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1   = GridAttentionBlock2D(    
                                                       in_channels=in_size, 
                                                       gating_channels=gate_size,
                                                       inter_channels=inter_size,
                                                       sub_sample_factor= sub_sample_factor
                                                  )
        self.combine_gates  =        nn.Sequential(
                                                        nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                                        nn.BatchNorm2d(in_size),
                                                        nn.ReLU(inplace=True)
                                                   )
        """
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')
        """
        
    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1



class GridAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, sub_sample_factor=(2,2,2)):
        super().__init__()
        self.upsample_mode     = 'bilinear'
        self.in_channels       = in_channels
        self.gating_channels   = gating_channels
        self.inter_channels    = inter_channels
        self.sub_sample_factor = sub_sample_factor
        
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta          = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                                        kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi            = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                                        kernel_size=1, stride=1, padding=0, bias=True)
        self.psi            = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Output transform
        self.W              = nn.Sequential(
                                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(self.in_channels),
                              )
        
    def forward(self, x, g) :
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f
