# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn.init import kaiming_normal_, calculate_gain

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


#################################################################################
# Construct Middle Classes ######################################################
#################################################################################
class PixelWiseNormLayer(nn.Module):
    """
    Mentioned in '4.2 PIXELWISE FEATURE VECTOR NORMALIZATION IN GENERATOR'
    'Local response normalization'
    """

    def __init__(self):
        super(PixelWiseNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="batch", conv_type="normal"):
        super(inconv, self).__init__()
        ##self.conv = double_conv(in_ch, out_ch)
        modules = []
        modules = add_conv(conv_type, modules, in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        modules = add_norm(norm_type, modules, out_ch)
        modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        modules = add_conv(conv_type, modules, out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        modules = add_norm(norm_type, modules, out_ch)
        modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="batch", batch_type = "batch", conv_type="normal", is_self_attn=False, is_deform_conv=False, recurrent_res=False, se_block=False):
        super(down, self).__init__()

        modules = []
        if recurrent_res:
            modules.append(RRCNN_block(in_ch, in_ch))

        modules.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        modules = add_conv(conv_type, modules, in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        modules = add_norm(norm_type, modules, out_ch)
        modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        modules = add_conv(conv_type, modules, out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        modules = add_norm(norm_type, modules, out_ch)
        modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        if is_self_attn:
            modules.append(self_attn(out_ch))
        if is_deform_conv:
            modules = add_conv("deform", modules, out_ch, out_ch, kernel_size=3, stride=1, padding=1)
            modules = add_norm(norm_type, modules, out_ch)
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        if se_block:
            modules.append(SEBlock(out_ch))

        self.mpconv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, use_dropout=False, norm_type="batch", conv_type="normal", is_self_attn=False, is_deform_conv=False, recurrent_res=False, se_block=False):
        super(up, self).__init__()

        modules = []
        if recurrent_res:
            modules.append(RRCNN_block(in_ch, in_ch))

        modules.append(nn.Upsample(scale_factor=2, mode='nearest'))

        modules = add_conv(conv_type, modules, in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        modules = add_norm(norm_type, modules, out_ch)
        modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))


        modules = add_conv(conv_type, modules, out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        modules = add_norm(norm_type, modules, out_ch)
        modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        if use_dropout:
            modules.append(nn.Dropout(0.5))

        if is_self_attn:
            modules.append(self_attn(out_ch))

        if is_deform_conv:
            modules = add_conv("deform", modules, out_ch, out_ch, kernel_size=3, stride=1, padding=1)
            modules = add_norm(norm_type, modules, out_ch)
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        if se_block:
            modules.append(SEBlock(out_ch))

        self.conv = nn.Sequential(*modules)

    def forward(self, x1, x2 = None):
        #x1 = self.up(x1)

        if x2 is not None:
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

            # for padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        x = self.conv(x)
        return x

class rgb_conv(nn.Module):
    def __init__(self, n_channels, n_classes, conv_type="normal"):
        super(rgb_conv, self).__init__()

        block_cache = []
        #block_cache.append(nn.ReLU(inplace=True))
        #block_cache.append(nn.ConvTranspose2d(n_channels, out_channels=n_classes, kernel_size=1, stride=1, padding=0, bias=False))
        block_cache = add_conv(conv_type, block_cache, n_channels, n_classes, kernel_size=1, stride=1, padding=0)

        #block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
        block_cache.append(nn.Tanh())

        self.rgb_layers_ = nn.Sequential(*block_cache)
    def forward(self, x):
        return self.rgb_layers_(x)

# Squeeze and Excitation Block Module
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()

        self.cse = cSEBlock(channel, reduction)
        self.sse = sSEBlock(channel, reduction)

    def forward(self, x):
        c = self.cse(x)
        s = self.sse(x)

        return c + s



# Squeeze and Excitation Block Module
class cSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(cSEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Squeeze and Excitation Block Module
class sSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(sSEBlock, self).__init__()

        self.cse = nn.Sequential(
            nn.Conv2d(channel, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cse(x)

class self_attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation="ReLU"):
        super(self_attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def add_conv(conv_type, block_cache, in_ch, out_ch, kernel_size=4, stride=1, padding=0, bias=False):
    if conv_type == "normal":
        block_cache.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    else:
        raise TypeError("'conv_type' must in ['normal', 'equal']")
    return block_cache

def add_norm(norm_type, block_cache, in_ch):
    if norm_type == "batch":
        block_cache.append(nn.BatchNorm2d(in_ch))
    elif norm_type == "c-batch":
        block_cache.append(ConditionalBatchNorm(in_ch))
    elif norm_type == "instance":
        block_cache.append(nn.InstanceNorm2d(in_ch))
    elif norm_type == "spectral":
        block_cache[-1] = nn.utils.spectral_norm(block_cache[-1])
    elif norm_type == "equal":
        block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
    elif norm_type == "pixel":
        block_cache.append(PixelwiseNorm())
    elif norm_type == "frn":
        block_cache.append(FilterResponseNorm2d(in_ch))
    elif norm_type == "none":
        return block_cache
    else:
        raise TypeError("'norm_type' must in ['batch', 'spectral', 'equal']")

    return block_cache

class ConditionalBatchNorm(nn.Module):
    def __init__(self, in_channel, n_condition=148):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)

        self.embed = nn.Linear(n_condition, in_channel* 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        # print(class_id.dtype)
        # print('class_id', class_id.size()) # torch.Size([4, 148])
        # print(out.size()) #torch.Size([4, 128, 4, 4])
        # class_id = torch.randn(4,1)
        # print(self.embed)
        embed = self.embed(class_id)
        # print('embed', embed.size())
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        # print(beta.size())
        out = gamma * out + beta

        return out

class _FilterResponseNorm(nn.Module):
    __constants__ = ["num_features", "eps", "eps_trainable", "tau", "beta", "gamma"]

    def __init__(self, shape, activated=True, eps=1e-6, eps_trainable=True):
        super(_FilterResponseNorm, self).__init__()
        self._eps = eps
        self.activated = activated
        self.num_features = shape[1]
        self.eps_trainable = eps_trainable

        self.beta = nn.Parameter(torch.zeros(shape))
        self.gamma = nn.Parameter(torch.ones(shape))

        if self.eps_trainable:
            self.eps = nn.Parameter(torch.full(shape, eps))
        else:
            self.eps = eps

        if self.activated:
            self.tau = nn.Parameter(torch.zeros(shape))
        else:
            self.tau = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)
        if isinstance(self.eps, nn.Parameter):
            nn.init.constant_(self.eps, self._eps)
        if self.tau is not None:
            nn.init.zeros_(self.tau)

    def _check_input_dim(self, input):
        raise NotImplementedError

class FilterResponseNorm2d(_FilterResponseNorm):

    def __init__(self, num_features, activated=True, eps=1e-6, eps_trainable=True):
        super(FilterResponseNorm2d, self).__init__(
            shape=(1, num_features, 1, 1),
            activated=activated,
            eps=eps,
            eps_trainable=eps_trainable,
        )

    def forward(self, input):
        self._check_input_dim(input)
        nu2 = torch.mean(input.pow(2), axis=[2, 3], keepdims=True)
        input = input * torch.rsqrt(nu2 + torch.abs(self.eps) + self._eps)
        output = self.gamma * input + self.beta
        if self.activated:
            output = torch.max(output, self.tau)
        return output

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class PixelwiseNorm(torch.nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
####### from pix2pix repo####

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, pg_level = -1):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.pg_level = pg_level
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:#or 0 == self.pg_level:
            return self.model(x)
        #elif 0 > self.pg_level:
        #    return x
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


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

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
