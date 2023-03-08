from __future__ import print_function
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
from itertools import cycle
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init
from torch.nn import Parameter
import numpy as np

class SDLayer(nn.Module):
    def __init__(self, num_input_channels=3,
                 out_channel=64, ks=7,stride=2, ista_iters=2,
                 lambd = 0.1):
        super(SDLayer, self).__init__()
        self._ista_iters = ista_iters
        self._layers = 1
        self.stride = stride
        self.kernelsize = ks
        self.out_channel = out_channel
        self.lamda = lambd
        self.softthrsh0 = None
        self.encode_conv0 = self.build_conv_layers(num_input_channels, self.out_channel, 1)[0]
        self.decode_deconv0 = self.build_deconv_layers(self.out_channel, num_input_channels, 1)[0]

    def build_softthrsh(self,_lambd):
        return SoftshrinkTrainable(
            Parameter(_lambd * torch.ones(1, self.out_channel), requires_grad=False)
        )

    def build_conv_layers(self,in_ch, out_ch, count):

        return nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, self.kernelsize,
                        stride=self.stride, padding=self.kernelsize // 2, bias=False) for _ in
                range(count)])

    def build_deconv_layers(self,in_ch, out_ch, count):
            #nn.ConvTranspose2d(64,3,kernel_size=7,stride=2,padding=3,output_padding=1,bias=False)
        return nn.ModuleList(
            [nn.ConvTranspose2d(in_ch, out_ch, self.kernelsize,
                        stride=self.stride, output_padding=1,padding=self.kernelsize // 2, bias=False) for _ in
                range(count)])
    @property
    def ista_iters(self):
        """Amount of ista iterations
        """
        return self._ista_iters

    @property
    def layers(self):
        """Amount of layers with free parameters.
        """
        return self._layers

    @property
    def conv_dictionary(self):
        """Get the weights of convolutoinal dictionary
        """
        return self.encode_conv0.weight.data

    @property
    def deconv_dictionary(self):
        """Get the weights of convolutoinal dictionary
        """
        return self.decode_deconv0.weight.data

    def init_z(self,x_shape):
        bs = x_shape[0]
        z_h = torch.floor(torch.tensor(((x_shape[2]+2*(self.kernelsize // 2)-(self.kernelsize-1)-1.0)//self.stride)+1))
        z_w = torch.floor(torch.tensor(((x_shape[3]+2*(self.kernelsize // 2)-(self.kernelsize-1)-1.0)//self.stride)+1))
        z = torch.zeros((bs,self.out_channel,int(z_h),int(z_w)))
        return z

    def init_t(self,deconv_weight):
        out_channel,in_channel,kernel_size1,kernel_size2=deconv_weight.shape
        A = deconv_weight.reshape(out_channel,-1)
        AA = A@torch.transpose(A,0,1)
        eigvalue = self.pow_iteration(AA)
        return 0.9/eigvalue


    def pow_iteration(self,A,max_iteration=50,threshold=1e-5):
        m = A.shape[0]
        eigenvector0 = torch.ones((m,1))
        eigenvector1 = A@eigenvector0
        k = 0
        while torch.abs(torch.max(eigenvector0)-torch.max(eigenvector1))>torch.sqrt(torch.tensor(threshold)) and k<max_iteration:
            eigenvector0 = eigenvector1
            eigenvector1 = A@(eigenvector0/torch.max(eigenvector0))
            k=k+1
        eigvalue = torch.max(eigenvector1)
        return eigvalue

    def iteration_z(self,t,inputs,z):
        self.softthrsh0 = self.build_softthrsh(self.lamda*t)
        z_pre = z
        m_pre = 1
        #print("===================>")
        #print(z_pre.shape)
        for i in range(self._ista_iters):
            previous_threshold = z_pre+t*self.encode_conv0(inputs - self.decode_deconv0(z_pre))
            z_pos = self.softthrsh0(previous_threshold)
            m_pos = (1+torch.sqrt(torch.tensor(1+4*m_pre*m_pre)))/2
            z_pre = z_pos+((m_pre-1)/m_pos)*(z_pos-z_pre)
            m_pre = m_pos
        return z_pre

    def forward(self, inputs):
        bs,ch,H,W = inputs.shape
        z0 = self.init_z(inputs.shape)
        t = self.init_t(self.decode_deconv0.weight.data)
        z = self.iteration_z(t,inputs,z0)
        #csc = self.forward_enc(inputs)
        #outputs = self.forward_dec(csc)
        return z

class SoftshrinkTrainable(nn.Module):
    """
    Learn threshold (lambda)
    """

    def __init__(self, _lambd):
        super(SoftshrinkTrainable, self).__init__()
        self._lambd = _lambd

    @property
    def thrshold(self):
        return self._lambd
#        self._lambd.register_hook(print)

    def forward(self, inputs):
        """ sign(inputs) * (abs(inputs)  - thrshold)"""
        _inputs = inputs
        _lambd = self._lambd.clamp(0).unsqueeze(-1).unsqueeze(-1)
        result = torch.sign(_inputs) * (F.relu(torch.abs(_inputs) - _lambd))
        return result

if __name__ =="__main__":
    layer = SDLayer(3,64, 7,2, 3,0.1)
    input = torch.randn((4,3,224,224))
    output = layer(input)
    convlayer = nn.Conv2d(3, 64, 7,
              2, 3, bias=False)
    relu = nn.ReLU()
    output2 = relu(convlayer(input))

    print(torch.sum(input < 0.1)/(4*3*224*224))
    print(torch.sum(output<0.1)/(4*64*112*112))
    print(torch.sum(output2 < 0.1) / (4 * 64 * 112 * 112))