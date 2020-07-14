import torch
import torch.nn as nn
from .discriminator_parts import *
import numpy as np

class ImageGAN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ImageGAN, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.outc = nn.Conv2d(1024, 1, kernel_size=4, stride=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.outc(x5)
        return x



class PG_MultiPatchDiscriminator(nn.Module):
    def __init__(self,
                 resolution,  # Output resolution. Overridden based on dataset.
                 input_channel=3,  # input channel size, for rgb always 3
                 fmap_base=2 ** 13,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=2 ** 9,  # Maximum number of feature maps in any layer.
                 is_sigmoid=False,
                 norm = "none",
                 conv_type = "normal",
                 use_pan=False,
                 se_block=False
                 ):
        super(PG_MultiPatchDiscriminator, self).__init__()

        self.big_model = PG_Discriminator(resolution=resolution, input_channel=input_channel, fmap_base=fmap_base,
                                          fmap_decay=fmap_decay, fmap_max=fmap_max, is_sigmoid=is_sigmoid, norm=norm,
                                          conv_type=conv_type, use_pan=use_pan, se_block=se_block).cuda()
        self.middle_model = PG_Discriminator(resolution=resolution / 2 / 2, input_channel=input_channel,
                                             fmap_base=fmap_base, fmap_decay=fmap_decay, fmap_max=fmap_max,
                                             is_sigmoid=is_sigmoid, norm=norm, conv_type=conv_type,
                                             use_pan=use_pan, se_block=se_block).cuda()
        self.small_model = PG_Discriminator(resolution=resolution / 2 / 2 / 2 / 2, input_channel=input_channel,
                                            fmap_base=fmap_base, fmap_decay=fmap_decay, fmap_max=fmap_max,
                                            is_sigmoid=is_sigmoid, norm=norm, conv_type=conv_type,
                                            use_pan=use_pan, se_block=se_block).cuda()

        self.downsample = nn.AvgPool2d(4, stride=4, count_include_pad=False)

    def set_config(self, resolution, status, alpha):
        self.big_model.set_config(resolution, status, alpha)
        self.middle_model.set_config(resolution / 2 / 2, status, alpha)
        self.small_model.set_config(resolution / 2 / 2 / 2 / 2, status, alpha)
        if status == "stable":
            self.current_resolution = resolution
        else:
            self.current_resolution = resolution / 2

    def forward(self, input, pop_intermediate=False):
        results = []
        results.append(self.big_model(input, pop_intermediate=pop_intermediate))
        results.append(self.middle_model(input, pop_intermediate=pop_intermediate))

        if self.current_resolution > 32:
            results.append(self.small_model(input, pop_intermediate=pop_intermediate))
        return results

    def get_intermediate_outputs(self):
        output = {}
        o = self.big_model.get_intermediate_outputs()
        for cuda_devices in o:
            for feature in o[cuda_devices]:
                if feature.device in output:
                    output[cuda_devices].append(feature)
                else:
                    output[cuda_devices] = [feature]

        o = self.middle_model.get_intermediate_outputs()
        for cuda_devices in o:
            for feature in o[cuda_devices]:
                output[cuda_devices].append(feature)

        if self.current_resolution > 32:
            o = self.small_model.get_intermediate_outputs()
            for cuda_devices in o:
                for feature in o[cuda_devices]:
                    output[cuda_devices].append(feature)
        return output

    def reset_intermediate_outputs(self):
        self.big_model.reset_intermediate_outputs()
        self.middle_model.reset_intermediate_outputs()
        self.small_model.reset_intermediate_outputs()


    def reset_intermediate_outputs(self):
        self.big_model.reset_intermediate_outputs()
        self.middle_model.reset_intermediate_outputs()
        self.small_model.reset_intermediate_outputs()


class PG_MultiScaleDiscriminator(nn.Module):
    def __init__(self,
                 resolution,  # Output resolution. Overridden based on dataset.
                 input_channel=3,  # input channel size, for rgb always 3
                 fmap_base=2 ** 13,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=2 ** 9,  # Maximum number of feature maps in any layer.
                 is_sigmoid=False,
                 norm = "none",
                 conv_type = "normal",
                 use_pan=False,
                 se_block=False
                 ):
        super(PG_MultiScaleDiscriminator, self).__init__()

        self.big_model = PG_Discriminator(resolution=resolution, input_channel=input_channel, fmap_base=fmap_base, fmap_decay=fmap_decay, fmap_max=fmap_max, is_sigmoid=is_sigmoid ,norm=norm, conv_type=conv_type, use_pan=use_pan, se_block=se_block).cuda()
        self.middle_model = PG_Discriminator(resolution=resolution/2/2, input_channel=input_channel, fmap_base=fmap_base, fmap_decay=fmap_decay, fmap_max=fmap_max, is_sigmoid=is_sigmoid,norm=norm, conv_type=conv_type, use_pan=use_pan, se_block=se_block).cuda()
        self.small_model = PG_Discriminator(resolution=resolution/2/2/2/2, input_channel=input_channel, fmap_base=fmap_base, fmap_decay=fmap_decay, fmap_max=fmap_max, is_sigmoid=is_sigmoid,norm=norm, conv_type=conv_type, use_pan=use_pan, se_block=se_block).cuda()

        self.downsample = nn.AvgPool2d(4, stride=4, count_include_pad=False)

    def set_config(self, resolution, status, alpha):
        self.big_model.set_config(resolution, status, alpha)
        self.middle_model.set_config(resolution/2/2, status, alpha)
        self.small_model.set_config(resolution/2/2/2/2, status, alpha)
        if status == "stable":
            self.current_resolution = resolution
        else:
            self.current_resolution = resolution/2

    def forward(self, input, pop_intermediate=False):
        results = []
        results.append(self.big_model(input, pop_intermediate=pop_intermediate))
        downsampled = self.downsample(input)
        results.append(self.middle_model(downsampled, pop_intermediate=pop_intermediate))

        if self.current_resolution > 32:
            downsampled = self.downsample(downsampled)
            results.append(self.small_model(downsampled, pop_intermediate=pop_intermediate))
        return results

    def get_intermediate_outputs(self):
        output = {}
        o = self.big_model.get_intermediate_outputs()
        for cuda_devices in o:
            for feature in o[cuda_devices]:
                if feature.device in output:
                    output[cuda_devices].append(feature)
                else:
                    output[cuda_devices] = [feature]

        o = self.middle_model.get_intermediate_outputs()
        for cuda_devices in o:
            for feature in o[cuda_devices]:
                output[cuda_devices].append(feature)

        o = self.small_model.get_intermediate_outputs()
        for cuda_devices in o:
            for feature in o[cuda_devices]:
                output[cuda_devices].append(feature)
        return output


    def reset_intermediate_outputs(self):
        self.big_model.reset_intermediate_outputs()
        self.middle_model.reset_intermediate_outputs()
        self.small_model.reset_intermediate_outputs()

class PG_Discriminator(nn.Module):
    def __init__(self,
                 resolution,  # Output resolution. Overridden based on dataset.
                 input_channel=3,  # input channel size, for rgb always 3
                 fmap_base=2 ** 13,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=2 ** 9,  # Maximum number of feature maps in any layer.
                 norm="none",
                 conv_type="equal",
                 is_sigmoid=False,
                 is_linear=False,
                 is_acgan=False,
                 is_msg=False,
                 add_noise=False,
                 is_self_attn=False,
                 use_pan=False,
                 is_deform_conv=False,
                 se_block = False
                 ):
        super(PG_Discriminator, self).__init__()
        self.input_channel_ = input_channel
        self.norm = norm
        self.conv_type = conv_type
        self.is_sigmoid_ = is_sigmoid
        self.is_linear = is_linear
        self.is_msg = is_msg
        self.is_self_attn = is_self_attn
        self.use_pan = use_pan

        self.add_noise = add_noise
        if self.add_noise:
            self.noise_layer=DynamicGNoise()
        self.is_deform_conv = is_deform_conv
        self.se_block = se_block

        # Use (fmap_max, fmap_decay, fmap_max)
        # to control every level's in||out channels
        self.fmap_base_ = fmap_base
        self.fmap_decay_ = fmap_decay
        self.fmap_max_ = fmap_max
        image_pyramid_ = int(np.log2(resolution))  # max level of the Image Pyramid
        self.resolution_ = 2 ** image_pyramid_  # correct resolution
        self.net_level_max_ = image_pyramid_ - 1  # minus 1 in order to exclude first rgb layer

        self.lod_layers_ = nn.ModuleList()  # layer blocks exclude to_rgb layer
        self.rgb_layers_ = nn.ModuleList()  # rgb layers each correspond to specific level.

        for level in range(self.net_level_max_, 0, -1):
            self._construct_by_level(level)

        self.net_level_ = self.net_level_max_  # set default net level as max level
        self.net_status_ = "stable"  # "stable" or "fadein"
        self.net_alpha_ = 1.0  # the previous stage's weight

        self.is_acgan = is_acgan
        if is_acgan:
            self.aux_layer = nn.Sequential(nn.AvgPool2d(8),
                                           View(-1, 128),
                                           nn.Linear(128, 128),
                                           nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                           nn.Linear(128, 8),
                                           nn.Softmax(dim=1)
                                           )

    @property
    def net_config(self):
        return self.net_level_, self.net_status_, self.net_alpha_

    @net_config.setter
    def net_config(self, config_list):
        self.net_level_, self.net_status_, self.net_alpha_ = config_list

    def set_config(self, resolution, status, alpha):
        self.net_level_, self.net_status_, self.net_alpha_ = [int(np.log2(resolution)) - 2, status, alpha]

    def forward(self, x, pop_intermediate=False):
        results = []

        if self.net_status_ == "stable":
            cur_input_level = self.net_level_max_ - self.net_level_ - 1
            x = self.rgb_layers_[cur_input_level](x)
            for cursor in range(cur_input_level, self.net_level_max_):
                if self.is_acgan and cursor == self.net_level_max_-2:
                    category = self.aux_layer(x)
                x = self.lod_layers_[cursor](x)
                results.append(x)

        elif self.net_status_ == "fadein":
            pre_input_level = self.net_level_max_ - self.net_level_
            cur_input_level = self.net_level_max_ - self.net_level_ - 1
            pre_weight, cur_weight = self.net_alpha_, 1.0 - self.net_alpha_
            x_pre_cache = self.rgb_layers_[pre_input_level](x)
            x_cur_cache = self.rgb_layers_[cur_input_level](x)
            x_cur_cache = self.lod_layers_[cur_input_level](x_cur_cache)
            x = HelpFunc.process_transition(x_pre_cache, x_cur_cache) * pre_weight + x_cur_cache * cur_weight

            for cursor in range(cur_input_level + 1, self.net_level_max_):
                if self.is_acgan and cursor == self.net_level_max_-2:
                    category = self.aux_layer(x)
                x = self.lod_layers_[cursor](x)
                results.append(x)

        else:
            raise AttributeError("Please set the net_status: ['stable', 'fadein']")

        return results

    def _construct_by_level(self, cursor):
        in_level = cursor
        out_level = cursor - 1
        in_channels, out_channels = map(self._get_channel_by_stage, (in_level, out_level))
        block_type = "Minibatch" if cursor == 1 else "DownSample"
        if self.is_msg and cursor != self.net_level_max_:
            inc = in_channels * 2
        else:
            inc = in_channels

        if self.is_deform_conv and cursor in [1, 2]:
            de_conv = True
        else:
            de_conv = False
        self._create_block(inc, out_channels, block_type, de_conv)  # construct (max_level-1) layers(exclude rgb layer)
        self._create_block(3, in_channels, "FromRGB")  # construct rgb layer for each previous level

    def _create_block(self, in_channels, out_channels, block_type, deform_conv=False):
        """
        Create a network block
        :param block_type:  only can be "Minibatch"||"DownSample"||"FromRGB"
        :return:
        """
        block_cache = []
        if block_type == "DownSample":
            if deform_conv:
                block_cache.append(DeformConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                            bias=False, modulation=True))
                block_cache.append(nn.BatchNorm2d(out_channels))
                block_cache.append(nn.LeakyReLU(negative_slope=0.2))

            block_cache = self.add_conv(block_cache, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

            block_cache = self.add_norm(block_cache, out_channels)
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))

            #block_cache.append(nn.Conv2d(out_channels, out_channels,
            #                             kernel_size=3, stride=1, padding=1, bias=False))
            block_cache = self.add_conv(block_cache, out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            block_cache = self.add_norm(block_cache, out_channels)
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            block_cache.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

            if self.se_block:
                block_cache.append(unet.unet_parts.SEBlock(out_channels))

            self.lod_layers_.append(nn.Sequential(*block_cache))
        elif block_type == "FromRGB":
            #block_cache.append(nn.Conv2d(in_channels= self.input_channel_, out_channels=out_channels,
            #                             kernel_size=1, stride=1, padding=0, bias=False))
            block_cache = self.add_conv(block_cache, self.input_channel_, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

            block_cache = self.add_norm(block_cache, out_channels)
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            self.rgb_layers_.append(nn.Sequential(*block_cache))
        elif block_type == "Minibatch":
            if self.is_self_attn:
                block_cache.append(unet.unet_parts.self_attn(in_channels))
            if deform_conv:
                block_cache.append(DeformConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                            bias=False, modulation=True))
                block_cache.append(nn.BatchNorm2d(in_channels))
                block_cache.append(nn.LeakyReLU(negative_slope=0.2))

            block_cache.append(MiniBatchAverageLayer())
            #block_cache.append(nn.Conv2d(in_channels + 1, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            block_cache = self.add_conv(block_cache, in_channels + 1, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

            block_cache = self.add_norm(block_cache, out_channels)
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            if self.is_self_attn:
                block_cache.append(unet.unet_parts.self_attn(out_channels))

            if self.is_linear:# use Sphere GAN
                #block_cache.append(nn.AvgPool2d(kernel_size=2))
                block_cache.append(View(-1, 64 * 8))
                block_cache.append(nn.Linear(64 * 8, 512))
            else:
                #block_cache.append(nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False))
                block_cache = self.add_conv(block_cache, out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False)
                block_cache = self.add_norm(block_cache, out_channels)
                block_cache.append(nn.LeakyReLU(negative_slope=0.2))

                #block_cache.append(nn.Conv2d(out_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False))
                block_cache = self.add_conv(block_cache, out_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
                #block_cache = self.add_norm(block_cache, out_channels)

                if self.is_sigmoid_ is True:
                    block_cache.append(nn.Sigmoid())
            self.lod_layers_.append(nn.Sequential(*block_cache))
        else:
            raise TypeError("'block_type' must in ['Minibatch', 'DownSample', 'FromRGB']")

    def _get_channel_by_stage(self, level):
        return min(int(self.fmap_base_ / (2.0 ** (level * self.fmap_decay_))), self.fmap_max_)

    def add_conv(self, block_cache, in_ch, out_ch, kernel_size=4, stride=1, padding=0, bias=False):
        if self.conv_type == "normal":
            block_cache.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif self.conv_type == "equal":
            block_cache.append(EqualConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding))
        else:
            raise TypeError("'norm_type' must in ['normal', 'equal']")

        return block_cache

    def add_norm(self, block_cache, in_ch):
        if self.norm == "batch":
            block_cache.append(nn.BatchNorm2d(in_ch))
        elif self.norm == "instance":
            block_cache.append(nn.InstanceNorm2d(in_ch))
        elif self.norm == "spectral":
            block_cache[-1] = nn.utils.spectral_norm(block_cache[-1])
        elif self.norm == "equal":
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
        elif self.norm == "none":
            return block_cache
        else:
            raise TypeError("'norm_type' must in ['batch', 'spectral', 'equal']")

        return block_cache

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


from torch.autograd import Function
class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply

class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

from math import sqrt
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module