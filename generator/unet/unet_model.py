# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import numpy as np

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, use_dropout=False, norm_type="batch",conv_type="equal", is_acgan=False, is_msg=False, is_self_attn=False,
                 is_deform_conv=False, recurrent_res=False, param_rate=1, se_block=False):
        super(UNet, self).__init__()
        self.is_msg = is_msg;

        self.inc = inconv(n_channels, 32*param_rate, norm_type=norm_type, conv_type=conv_type)

        self.down0 = down(32*param_rate, 64*param_rate, conv_type=conv_type, norm_type=norm_type, recurrent_res=recurrent_res, se_block=se_block)
        self.down1 = down(64*param_rate, 64*param_rate, conv_type=conv_type, norm_type=norm_type, recurrent_res=recurrent_res, se_block=se_block)
        self.down2 = down(64*param_rate, 64*param_rate, conv_type=conv_type, norm_type=norm_type, is_self_attn=is_self_attn, is_deform_conv=is_deform_conv, recurrent_res=recurrent_res, se_block=se_block)
        self.down3 = down(64*param_rate, 128*param_rate, conv_type=conv_type, norm_type=norm_type, is_self_attn=is_self_attn, is_deform_conv=is_deform_conv, recurrent_res=recurrent_res, se_block=se_block)
        self.down4 = down(128*param_rate, 128*param_rate, conv_type=conv_type, norm_type=norm_type, recurrent_res=recurrent_res, se_block=se_block)
        self.down5 = down(128*param_rate, 128*param_rate, conv_type=conv_type, norm_type=norm_type, recurrent_res=recurrent_res, se_block=se_block)
        self.down6 = down(128*param_rate, 128*param_rate, conv_type=conv_type, norm_type=norm_type, recurrent_res=recurrent_res, se_block=se_block)
        #self.down7 = down(128, 128, batch_type="instance")

        if is_acgan:
            in_dim = 256
        else:
            in_dim = 128
        self.up0 = up(in_dim*param_rate, 128*param_rate, use_dropout=use_dropout, conv_type=conv_type, norm_type=norm_type, recurrent_res=recurrent_res, se_block=se_block)# bottom

        #self.up1 = up(256, 128)
        self.up2 = up(256*param_rate, 128*param_rate, use_dropout=use_dropout, conv_type=conv_type, norm_type=norm_type, recurrent_res=recurrent_res, se_block=se_block)
        self.up3 = up(256*param_rate, 128*param_rate, use_dropout=use_dropout, conv_type=conv_type, norm_type=norm_type, recurrent_res=recurrent_res, se_block=se_block)
        self.up4 = up(256*param_rate, 64*param_rate, conv_type=conv_type, norm_type=norm_type, is_self_attn=is_self_attn, is_deform_conv=is_deform_conv, recurrent_res=recurrent_res, se_block=se_block)
        self.up5 = up(128*param_rate, 64*param_rate, conv_type=conv_type, norm_type=norm_type, is_self_attn=is_self_attn, is_deform_conv=is_deform_conv, recurrent_res=recurrent_res, se_block=se_block)
        self.up6 = up(128*param_rate, 64*param_rate, conv_type=conv_type, norm_type=norm_type, recurrent_res=recurrent_res, se_block=se_block)
        self.up7 = up(128*param_rate, 32*param_rate, conv_type=conv_type, norm_type=norm_type, recurrent_res=recurrent_res, se_block=se_block)

        self.rgb_layer4 = rgb_conv(n_channels = 128*param_rate, n_classes = 3, conv_type=conv_type)
        self.rgb_layer8 = rgb_conv(n_channels = 128*param_rate, n_classes = 3, conv_type=conv_type)
        self.rgb_layer16 = rgb_conv(n_channels = 128*param_rate, n_classes = 3, conv_type=conv_type)
        self.rgb_layer32 = rgb_conv(n_channels = 64*param_rate, n_classes = 3, conv_type=conv_type)
        self.rgb_layer64 = rgb_conv(n_channels = 64*param_rate, n_classes = 3, conv_type=conv_type)
        self.rgb_layer128 = rgb_conv(n_channels = 64*param_rate, n_classes = 3, conv_type=conv_type)
        self.rgb_layer256 = rgb_conv(n_channels = 32*param_rate, n_classes = 3, conv_type=conv_type)

        self.is_acgan = is_acgan
        if self.is_acgan:
            self.label_emb = nn.Sequential(
                nn.Embedding(8, 50),
                nn.Linear(50, 64),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.z = torch.from_numpy(np.random.normal(0, 1, (6, 100))).float().cuda()

    def forward(self, x, label=None):
        in256 = self.inc(x)

        x1 = self.down0(in256)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        #x8 = self.down7(x7)

        if self.is_acgan:
            lab = self.label_emb(label)
            #print(lab.shape)
            lab = lab.view(lab.shape[0], 128, 2, 2)
            #print(x7.shape)
            x7 = torch.cat([x7, lab], 1)
            #gen_input = torch.mul(self.label_emb(label), self.z)
            #emb = emb.reshape(x7.shape)
            #print(x7.shape)
            #x7 = torch.einsum('nhwc,nc->nhw', x7, gen_input)
            #x7 = torch.mul(emb, x7)
        x7 = self.up0(x7)# bottom
        if self.is_msg: msg_result = []
        if self.net_status_ == "stable" or self.is_msg:
            #if(self.net_level_ == 0):return self.outc4(x)
            #x = self.up1(x8, x7)
            #if(self.net_level_ == 1):return self.rgb_layer(x), x16
            if (self.is_msg): msg_result.append(self.rgb_layer4(x7))
            x = self.up2(x7, x6)
            if (self.is_msg): msg_result.append(self.rgb_layer8(x))
            elif(self.net_level_ == 2): return self.rgb_layer8(x)
            x = self.up3(x, x5)
            #x16 = self.rgb_layer(x).clone()
            if (self.is_msg): msg_result.append(self.rgb_layer16(x))
            elif(self.net_level_ == 3): return self.rgb_layer16(x)
            x = self.up4(x, x4)
            if (self.is_msg): msg_result.append(self.rgb_layer32(x))
            elif(self.net_level_ == 4): return self.rgb_layer32(x)
            x = self.up5(x, x3)
            if (self.is_msg): msg_result.append(self.rgb_layer64(x))
            elif(self.net_level_ == 5): return self.rgb_layer64(x)
            x = self.up6(x, x2)
            if (self.is_msg): msg_result.append(self.rgb_layer128(x))
            elif(self.net_level_ == 6): return self.rgb_layer128(x)
            x = self.up7(x, x1)
            if (self.is_msg): msg_result.append(self.rgb_layer256(x))
            else: return self.rgb_layer256(x)

        elif self.net_status_ == "fadein":
            pre_output_level = self.net_level_ - 1
            pre_weight, cur_weight = self.net_alpha_, 1.0 - self.net_alpha_

            output_cache = []
            #x = self.up1(x8, x7)
            #if (self.net_level_ == 1 or self.net_level_ == 2):
            #    output_cache.append(self.rgb_layer(x))

            if (len(output_cache) < 2): x = self.up2(x7, x6)
            if (self.net_level_ == 2 or self.net_level_ == 3):
                output_cache.append(self.rgb_layer8(x))

            if (len(output_cache) < 2): x = self.up3(x, x5)
            #x16 = self.rgb_layer(x).clone()
            if (self.net_level_ == 3 or self.net_level_ == 4):
                output_cache.append(self.rgb_layer16(x))

            if (len(output_cache) < 2): x = self.up4(x, x4)
            if (self.net_level_ == 4 or self.net_level_ == 5):
                output_cache.append(self.rgb_layer32(x))

            if (len(output_cache) < 2): x = self.up5(x, x3)
            if (self.net_level_ == 5 or self.net_level_ == 6):
                output_cache.append(self.rgb_layer64(x))

            if (len(output_cache) < 2): x = self.up6(x, x2)
            if (self.net_level_ == 6 or self.net_level_ == 7):
                output_cache.append(self.rgb_layer128(x))

            if (len(output_cache) < 2): x = self.up7(x, x1)
            if (self.net_level_ == 7):
                output_cache.append(self.rgb_layer256(x))
            x = HelpFunc.process_transition(output_cache[0], output_cache[1]) * pre_weight \
                + output_cache[1] * cur_weight
            return x

        if self.is_msg: return msg_result
        else: return x

    def set_config(self, resolution, status, alpha):
        self.net_level_, self.net_status_, self.net_alpha_ = [int(np.log2(resolution)) - 1, status, alpha]



#################################################################################
# Construct Generator and Discriminator #########################################
#################################################################################
class PGGenerator(nn.Module):
    def __init__(self,
                 resolution,  # Output resolution. Overridden based on dataset.
                 latent_size,  # Dimensionality of the latent vectors.
                 final_channel=3,  # Output channel size, for rgb always 3
                 fmap_base=2 ** 13,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=2 ** 9,  # Maximum number of feature maps in any layer.
                 is_tanh=False
                 ):
        super(PGGenerator, self).__init__()
        self.latent_size_ = latent_size
        self.is_tanh_ = is_tanh
        self.final_channel_ = final_channel
        # Use (fmap_max, fmap_decay, fmap_max)
        # to control every level's in||out channels
        self.fmap_base_ = fmap_base
        self.fmap_decay_ = fmap_decay
        self.fmap_max_ = fmap_max
        image_pyramid_ = int(np.log2(resolution))  # max level of the Image Pyramid
        self.resolution_ = 2 ** image_pyramid_  # correct resolution
        self.net_level_max_ = image_pyramid_ - 1  # minus 1 in order to exclude last rgb layer

        self.lod_layers_ = nn.ModuleList()    # layer blocks exclude to_rgb layer
        self.rgb_layers_ = nn.ModuleList()    # rgb layers each correspond to specific level.

        for level in range(self.net_level_max_):
            self._construct_by_level(level)

        self.net_level_ = self.net_level_max_  # set default net level as max level
        self.net_status_ = "stable"            # "stable" or "fadein"
        self.net_alpha_ = 1.0                  # the previous stage's weight

    @property
    def net_config(self):
        """
        Return current net's config.
        The config is used to control forward
        The pipeline was mentioned below Figure2 of the Paper
        """
        return self.net_level_, self.net_status_, self.net_alpha_

    @net_config.setter
    def net_config(self, config_list):
        """
        :param iterable config_list: [net_level, net_status, net_alpha]
        :return:
        """
        self.net_level_, self.net_status_, self.net_alpha_ = config_list

    def forward(self, x):
        """
        The pipeline was mentioned below Figure2 of the Paper
        """
        if self.net_status_ == "stable":
            cur_output_level = self.net_level_
            for cursor in range(self.net_level_+1):
                x = self.lod_layers_[cursor](x)
            x = self.rgb_layers_[cur_output_level](x)

        elif self.net_status_ == "fadein":
            pre_output_level = self.net_level_ - 1
            cur_output_level = self.net_level_
            pre_weight, cur_weight = self.net_alpha_, 1.0 - self.net_alpha_
            output_cache = []
            for cursor in range(self.net_level_+1):
                x = self.lod_layers_[cursor](x)
                if cursor == pre_output_level:
                    output_cache.append(self.rgb_layers_[cursor](x))
                if cursor == cur_output_level:
                    output_cache.append(self.rgb_layers_[cursor](x))
            x = HelpFunc.process_transition(output_cache[0], output_cache[1]) * pre_weight \
                + output_cache[1] * cur_weight

        else:
            raise AttributeError("Please set the net_status: ['stable', 'fadein']")

        return x

    def _construct_by_level(self, cursor):
        in_level = cursor
        out_level = cursor + 1
        in_channels, out_channels = map(self._get_channel_by_stage, (in_level, out_level))
        block_type = "First" if cursor == 0 else "UpSample"
        self._create_block(in_channels, out_channels, block_type)  # construct previous (max_level - 1) layers
        self._create_block(out_channels, 3, "ToRGB")                # construct rgb layer for each previous level

    def _create_block(self, in_channels, out_channels, block_type):
        """
        Create a network block
        :param block_type:  only can be "First"||"UpSample"||"ToRGB"
        :return:
        """
        block_cache = []
        if block_type in ["First", "UpSample"]:
            if block_type == "First":
                block_cache.append(PixelWiseNormLayer())
                block_cache.append(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=4, stride=1, padding=3, bias=False))
            if block_type == "UpSample":
                block_cache.append(nn.Upsample(scale_factor=2, mode='nearest'))
                block_cache.append(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, stride=1, padding=1, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            block_cache.append(PixelWiseNormLayer())
            block_cache.append(nn.Conv2d(out_channels, out_channels,
                                         kernel_size=3, stride=1, padding=1, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            block_cache.append(PixelWiseNormLayer())
            self.lod_layers_.append(nn.Sequential(*block_cache))
        elif block_type == "ToRGB":
            block_cache.append(nn.Conv2d(in_channels, out_channels=3,
                                         kernel_size=1, stride=1, padding=0, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            if self.is_tanh_ is True:
                block_cache.append(nn.Tanh())
            self.rgb_layers_.append(nn.Sequential(*block_cache))
        else:
            raise TypeError("'block_type' must in ['First', 'UpSample', 'ToRGB']")

    def _get_channel_by_stage(self, level):
        return min(int(self.fmap_base_ / (2.0 ** (level * self.fmap_decay_))), self.fmap_max_)


####from pix2pix repo ###

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer="batch", use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()

        norm_layer = get_norm_layer(norm_layer)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, pg_level = 0)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, pg_level = 1)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, pg_level = 2)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, pg_level = 3)  # add the outermost layer
        print(self.model)
        #print(self.model[10])
        self.rgb_layers_ = nn.ModuleList()  # rgb layers each correspond to specific level.
        self._create_rgb_layer()
        self.pg_level = 0
        #model.
    def forward(self, input):
        """Standard forward"""
        o = self.model[0]

        m_l = len(self.model)
        for m in range(1, m_l-10):
            o = self.model[m](o)
        #m = self.model(input)
        return o

    def _create_rgb_layer(self):
        block_cache = []
        block_cache.append(nn.Conv2d(3 , out_channels=3,
                                     kernel_size=1, stride=1, padding=0, bias=False))
        block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
        #if self.is_tanh_ is True:
        block_cache.append(nn.Tanh())
        self.rgb_layers_.append(nn.Sequential(*block_cache))

    def set_config(self, resolution, status, alpha):
        self.net_level_, self.net_status_, self.net_alpha_ = [int(np.log2(resolution)) - 3, status, alpha]