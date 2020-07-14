import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, calculate_gain
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_ch,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).cuda()

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

from torch.autograd import Variable
class DynamicGNoise(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std
        self.noise = Variable(torch.tensor(0.0).cuda())

    def forward(self, x):
        if not self.training: return x
        if x.shape != self.noise.shape:
            self.noise = Variable(torch.tensor(0.0).expand_as(x).cuda())
        self.noise.data.normal_(0, std=self.std)

        #print(x.size(), self.noise.size())
        return x + self.noise
#################################################################################
# Construct Help Functions Class#################################################
#################################################################################
class HelpFunc(object):
    @staticmethod
    def process_transition(a, b):
        """
        Transit tensor a as tensor b's size by
        'nearest neighbor filtering' and 'average pooling' respectively
        which mentioned below Figure2 of the Paper https://arxiv.org/pdf/1710.10196.pdf
        :param torch.Tensor a: is a tensor with size [batch, channel, height, width]
        :param torch.Tensor b: similar as a
        :return torch.Tensor :
        """
        a_batch, a_channel, a_height, a_width = a.size()
        b_batch, b_channel, b_height, b_width = b.size()
        # Drop feature maps
        if a_channel > b_channel:
            a = a[:, :b_channel]

        if a_height > b_height:
            assert a_height % b_height == 0 and a_width % b_width == 0
            assert a_height / b_height == a_width / b_width
            ks = int(a_height // b_height)
            a = F.avg_pool2d(a, kernel_size=ks, stride=ks, padding=0, ceil_mode=False, count_include_pad=False)

        if a_height < b_height:
            assert b_height % a_height == 0 and b_width % a_width == 0
            assert b_height / a_height == b_width / a_width
            sf = b_height // a_height
            a = F.interpolate(a, scale_factor=sf, mode='nearest')

        # Add feature maps.
        if a_channel < b_channel:
            z = torch.zeros((a_batch, b_channel - a_channel, b_height, b_width))
            a = torch.cat([a, z], 1)
        # print("a size: ", a.size())
        return a
class EqualizedLearningRateLayer(nn.Module):
    """
    Mentioned in '4.1 EQUALIZED LEARNING RATE'
    Applies equalized learning rate to the preceding layer.
    *'To initialize all bias parameters to zero and all weights
    according to the normal distribution with unit variance'
    """

    def __init__(self, layer):
        super(EqualizedLearningRateLayer, self).__init__()
        self.layer_ = layer

        # He's Initializer (He et al., 2015)
        kaiming_normal_(self.layer_.weight, a=calculate_gain('conv2d'))
        # Cause mean is 0 after He-kaiming function
        self.layer_norm_constant_ = (torch.mean(self.layer_.weight.data ** 2)) ** 0.5
        self.layer_.weight.data.copy_(self.layer_.weight.data / self.layer_norm_constant_)

        self.bias_ = self.layer_.bias if self.layer_.bias else None
        self.layer_.bias = None

    def forward(self, x):
        self.layer_norm_constant_ = self.layer_norm_constant_.type(torch.cuda.FloatTensor)
        x = self.layer_norm_constant_ * x
        if self.bias_ is not None:
            # x += self.bias.view(1, -1, 1, 1).expand_as(x)
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x

class MiniBatchAverageLayer(nn.Module):
    def __init__(self,
                 offset=1e-8  # From the original implementation
                              # https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py #L135
                 ):
        super(MiniBatchAverageLayer, self).__init__()
        self.offset_ = offset

    def forward(self, x):
        # Follow Chapter3 of the Paper:
        # Computer the standard deviation for each feature
        # in each spatial locations to arrive at the single value
        stddev = torch.sqrt(torch.mean((x - torch.mean(x, dim=0, keepdim=True))**2, dim=0, keepdim=True) + self.offset_)
        inject_shape = list(x.size())[:]
        inject_shape[1] = 1  #  Inject 1 line data for the second dim (channel dim). See Chapter3 and Table2
        inject = torch.mean(stddev, dim=1, keepdim=True)
        inject = inject.expand(inject_shape)
        return torch.cat((x, inject), dim=1)

