from torch import nn
from timm.layers.weight_init import trunc_normal_, lecun_normal_
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
import numpy as np

from functools import partial

from torch.nn.init import calculate_gain, _calculate_correct_fan
import math


def default_init_(m):
    # default init for Linear or Conv in Pytorch based on Kaiming Uniform
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def trunc_init_(m, std=0.02):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def trunc_weight_init_(w):
    trunc_normal_(w, std=.02)




def xavier_normal_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # elif isinstance(m, nn.Embedding):
    #     # nn.Embedding, fan_in is assumeed as 1 for all embedding vectors) due to the one-hot
    #     nn.init.normal_(m.weight, mean=0, std=np.sqrt(2/(1+m.weight.size(1))))



def xavier_uniform_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        # nn.Embedding, fan_in is assumeed as 1 for all embedding vectors) due to the one-hot
        scale = np.sqrt(6/(m.weight.size(1)+1))
        nn.init.uniform_(m.weight, -scale, scale)



# def trunc_init_(m):
#     if isinstance(m, (nn.Linear, nn.Conv1d)):
#         trunc_normal_(m.weight, std=0.02)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)


def trunc_normal_fan_init_(m, std=0.02, mode='fan_in'):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        fan = _calculate_correct_fan(m.weight, mode)
        trunc_normal_(m.weight, 0, std=std/np.sqrt(fan))
        if m.bias is not None:
            nn.init.zeros_(m.bias)





def uniform_init_(m, a=-1, b=1):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.uniform_(m.weight, a, b)
        if m.bias is not None:
            nn.init.zeros_(m.bias)



def kaiming_uniform_init_(m, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_uniform_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        # nn.Embedding, fan_in is assumeed as 1 for all embedding vectors) due to the one-hot
        fan = 1 if mode=='fan_in' else m.weight.size(1)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.) * std
        nn.init.uniform_(m.weight, -bound, bound)


kaiming_uniform_linear_init_ = partial(kaiming_uniform_init_, a=1)


def kaiming_normal_init_(m, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def kaiming_normal_clamp_init_(m, a=0, mode='fan_in', nonlinearity='leaky_relu', min=-2., max=2.):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        m.weight.clamp_(min=-2., max=2.)
        if m.bias is not None:
            nn.init.zeros_(m.bias)




def kaiming_normal_linear_init_(m, mode='fan_in'):
    '''
        Kaiming Guassian initialization for no-activation (i.e., leaky_relu with negative slope as 1.
    '''
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight, a=1., mode=mode, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def lecun_normal_init_(m):
    '''
        Kaiming Guassian initialization for no-activation (i.e., leaky_relu with negative slope as 1.
    '''
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def apply_weight_norm(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        weight_norm(m)



def apply_spectral_norm(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        spectral_norm(m)

