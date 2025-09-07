import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
		Args:
		  x: (Variable) top feature map to be upsampled.
		  y: (Variable) lateral feature map.
		Returns:
		  (Variable) added feature map.
		Note in PyTorch, when input size is odd, the upsampled feature map
		with `F.upsample(..., scale_factor=2, mode='nearest')`
		maybe not equal to the lateral feature map size.
		e.g.
		original input size: [N,_,15,15] ->
		conv2d feature map size: [N,_,8,8] ->
		upsampled feature map size: [N,_,16,16]
		So we choose bilinear upsample which supports arbitrary output sizes.
		'''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, rc1 =None,rc2=None , rc3=None , subnets=None):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        if rc1 != None and rc2!= None and rc3!=None: 
            c1 = subnets[0](rc1.to("cuda") , c1.to("cuda"))
            c2 = subnets[1](rc2.to("cuda") , c2.to("cuda"))
            c3 = subnets[2](rc3.to("cuda") , c3.to("cuda"))
            # c1 = h1 + c1*0.6
            # c2 = h2 + c2*0.6
            # c3 = h3 + c3*0.6

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out





# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

# from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
# from models.stylegan2.model import EqualLinear


# class GradualStyleBlock(Module):
#     def __init__(self, in_c, out_c, spatial):
#         super(GradualStyleBlock, self).__init__()
#         self.out_c = out_c
#         self.spatial = spatial
#         num_pools = int(np.log2(spatial))
#         modules = []
#         modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
#         for i in range(num_pools - 1):
#             modules += [
#                 Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()
#             ]
#         self.convs = nn.Sequential(*modules)
#         self.linear = EqualLinear(out_c, out_c, lr_mul=1)

#     def forward(self, x):
#         x = self.convs(x)
#         x = x.view(-1, self.out_c)
#         x = self.linear(x)
#         return x


# class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
		Args:
		  x: (Variable) top feature map to be upsampled.
		  y: (Variable) lateral feature map.
		Returns:
		  (Variable) added feature map.
		Note in PyTorch, when input size is odd, the upsampled feature map
		with `F.upsample(..., scale_factor=2, mode='nearest')`
		maybe not equal to the lateral feature map size.
		e.g.
		original input size: [N,_,15,15] ->
		conv2d feature map size: [N,_,8,8] ->
		upsampled feature map size: [N,_,16,16]
		So we choose bilinear upsample which supports arbitrary output sizes.
		'''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, rc1 =None,rc2=None , rc3=None, subnets=None):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # if rc1 != None and rc2!= None and rc3!=None: 
        #     # # h1 = 0.1 * torch.matmul(c1,rc1)
        #     # # h2 = 0.1 *torch.matmul(c2,rc2)
        #     # h3 = 0.2 * torch.matmul(c3,rc3)
        #     # # c1 = h1 + c1
        #     # # c2 = h2 + c2
        #     # c3 = h3 + c3
        #     h1 = subnets[0](rc1.to("cuda") , c1.to("cuda"))
        #     h2 = subnets[1](rc2.to("cuda") , c2.to("cuda"))
        #     h3 = subnets[2](rc3.to("cuda") , c3.to("cuda"))
        #     print(1)
        #     # print(f"{h2=}")
        #     # print(f"{h3=}")
        #     # c1 = h1 
        #     # c2 = h2 
        #     # c3 = h3 
        #     c1 = h1 + c1
        #     c2 = h2 + c2
        #     c3 = h3 + c3

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out