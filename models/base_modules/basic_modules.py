from utils.pipeline_utils import *
import consts
from tqdm import tqdm

import logging
import random
from collections import OrderedDict
import imageio
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits_loss
from torch.optim import Adam
from torch.utils.data import DataLoader

import models.stylegan2.model
from models.encoders import psp_encoders
from utils import common, train_utils
from criteria import id_loss, moco_loss
from configs import data_configs
from criteria.lpips.lpips import LPIPS

torch.autograd.set_detect_anomaly(True)
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        num_conv_layers = 6

        self.conv_layers = nn.ModuleList()

        def add_conv(module_list, name, in_ch, out_ch, kernel, stride, padding, act_fn):
            return module_list.add_module(
                name,
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride,
                    ),
                    act_fn,
                ),
            )

        add_conv(
            self.conv_layers,
            "e_conv_1",
            in_ch=3,
            out_ch=64,
            kernel=5,
            stride=2,
            padding=2,
            act_fn=nn.ReLU(),
        )
        add_conv(
            self.conv_layers,
            "e_conv_2",
            in_ch=64,
            out_ch=128,
            kernel=5,
            stride=2,
            padding=2,
            act_fn=nn.ReLU(),
        )
        add_conv(
            self.conv_layers,
            "e_conv_3",
            in_ch=128,
            out_ch=256,
            kernel=5,
            stride=2,
            padding=2,
            act_fn=nn.ReLU(),
        )
        add_conv(
            self.conv_layers,
            "e_conv_4",
            in_ch=256,
            out_ch=512,
            kernel=5,
            stride=2,
            padding=2,
            act_fn=nn.ReLU(),
        )
        add_conv(
            self.conv_layers,
            "e_conv_5",
            in_ch=512,
            out_ch=1024,
            kernel=5,
            stride=2,
            padding=2,
            act_fn=nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "e_fc_1",
                        nn.Linear(in_features=1024, out_features=consts.NUM_Z_CHANNELS),
                    ),
                    ("tanh_1", nn.Tanh()),  # normalize to [-1, 1] range
                ]
            )
        )

    def forward(self, face):
        out = face
        for conv_layer in self.conv_layers:
            # print("H")
            out = conv_layer(out)
            # print(out.shape)
            # print("W")
        out = out.flatten(1, -1)
        out = self.fc_layer(out)
        return out


class DiscriminatorZ(nn.Module):
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        dims = (
            512,
            consts.NUM_ENCODER_CHANNELS,
            consts.NUM_ENCODER_CHANNELS // 2,
            consts.NUM_ENCODER_CHANNELS // 4,
        )
        # dims = (consts.NUM_Z_CHANNELS, consts.NUM_ENCODER_CHANNELS, consts.NUM_ENCODER_CHANNELS // 2,
        #         consts.NUM_ENCODER_CHANNELS // 4)
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.layers.add_module(
                "dz_fc_%d" % i,
                nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
                ),
            )

        self.layers.add_module(
            "dz_fc_%d" % (i + 1),
            nn.Sequential(
                nn.Linear(out_dim, 1),
                # nn.Sigmoid()  # commented out because logits are needed
            ),
        )

    def forward(self, z):
        out = z
        for layer in self.layers:
            out = layer(out)
        return out


class DiscriminatorZ_multistyle(nn.Module):
    def __init__(self):
        super(DiscriminatorZ_multistyle, self).__init__()
        channel = consts.NUM_AGES
        dims = []
        while channel != 1:
            dims.append(channel)
            channel //= 2
        self.layers = nn.ModuleList()
        for i, (in_channel, out_channel) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.layers.add_module(
                f"dz_conv1d_{i}",
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channel, out_channels=out_channel, kernel_size=3
                    ),
                    nn.BatchNorm1d(num_features=out_channel),
                    nn.ReLU(),
                ),
            )
        self.layers.add_module(
            f"dz_conv1d_{i+1}",
            nn.Sequential(
                nn.Conv1d(in_channels=out_channel, out_channels=1, kernel_size=3)
            ),
        )
        self.fc_layers = nn.ModuleList()
        self.fc_layers.add_module(
            "dz_fc_1", nn.Linear(in_features=506, out_features=506 // 2)
        )
        self.fc_layers.add_module(
            "dz_fc_2", nn.Linear(in_features=506 // 2, out_features=1)
        )

    def forward(self, z: torch.tensor):
        out: torch.tensor = z
        for layer in self.layers:
            out = layer(out)
            # print(out.shape)

        out = out.flatten(1, -1)

        for layer in self.fc_layers:
            out = layer(out)
            # print(out.shape)
        return out


class DiscriminatorImg(nn.Module):
    def __init__(self):
        super(DiscriminatorImg, self).__init__()
        # in_dims = (3, 16 + consts.LABEL_LEN_EXPANDED, 32, 64)
        in_dims = (3, 118, 32, 64)
        out_dims = (16, 32, 64, 128)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims), 1):
            self.conv_layers.add_module(
                "dimg_conv_%d" % i,
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(),
                ),
            )

        self.fc_layers.add_module(
            "dimg_fc_1", nn.Sequential(nn.Linear(128 * 8 * 8, 1024), nn.LeakyReLU())
        )

        self.fc_layers.add_module(
            "dimg_fc_2",
            nn.Sequential(
                nn.Linear(1024, 1),
                # nn.Sigmoid()  # commented out because logits are needed
            ),
        )

    def forward(self, imgs, labels, device):
        out = imgs

        # run convs
        for i, conv_layer in enumerate(self.conv_layers, 1):
            # print(out.shape)
            # print(conv_layer)
            out = conv_layer(out)
            if i == 1:
                # concat labels after first conv
                labels_tensor = torch.zeros(
                    torch.Size((out.size(0), labels.size(1), out.size(2), out.size(3))),
                    device=device,
                )
                for img_idx in range(out.size(0)):
                    for label in range(labels.size(1)):
                        labels_tensor[img_idx, label, :, :] = labels[
                            img_idx, label
                        ]  # fill a square with either 1(for label) or 0(otherwise)
                out = torch.cat((out, labels_tensor), 1)  # important for concatenation

        # run fcs
        out = out.flatten(1, -1)
        for fc_layer in self.fc_layers:
            # print(out.shape)
            # print(fc_layer)

            out = fc_layer(out)

        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        num_deconv_layers = 5
        mini_size = 4
        self.fc = nn.Sequential(
            nn.Linear(
                consts.NUM_Z_CHANNELS + consts.LABEL_LEN_EXPANDED,
                consts.NUM_GEN_CHANNELS * mini_size**2,
            ),
            nn.ReLU(),
        )
        # need to reshape now to ?,1024,8,8

        self.deconv_layers = nn.ModuleList()

        def add_deconv(name, in_dims, out_dims, kernel, stride, actf):
            self.deconv_layers.add_module(
                name,
                nn.Sequential(
                    easy_deconv(
                        in_dims=in_dims,
                        out_dims=out_dims,
                        kernel=kernel,
                        stride=stride,
                    ),
                    actf,
                ),
            )

        add_deconv(
            "g_deconv_1",
            in_dims=(1024, 4, 4),
            out_dims=(512, 8, 8),
            kernel=5,
            stride=2,
            actf=nn.ReLU(),
        )
        add_deconv(
            "g_deconv_2",
            in_dims=(512, 8, 8),
            out_dims=(256, 16, 16),
            kernel=5,
            stride=2,
            actf=nn.ReLU(),
        )
        add_deconv(
            "g_deconv_3",
            in_dims=(256, 16, 16),
            out_dims=(128, 32, 32),
            kernel=5,
            stride=2,
            actf=nn.ReLU(),
        )
        add_deconv(
            "g_deconv_4",
            in_dims=(128, 32, 32),
            out_dims=(64, 64, 64),
            kernel=5,
            stride=2,
            actf=nn.ReLU(),
        )
        add_deconv(
            "g_deconv_5",
            in_dims=(64, 64, 64),
            out_dims=(32, 128, 128),
            kernel=5,
            stride=2,
            actf=nn.ReLU(),
        )
        add_deconv(
            "g_deconv_6",
            in_dims=(32, 128, 128),
            out_dims=(16, 128, 128),
            kernel=5,
            stride=1,
            actf=nn.ReLU(),
        )
        add_deconv(
            "g_deconv_7",
            in_dims=(16, 128, 128),
            out_dims=(3, 128, 128),
            kernel=1,
            stride=1,
            actf=nn.Tanh(),
        )

    def _decompress(self, x):
        return x.view(x.size(0), 1024, 4, 4)  # TODO - replace hardcoded

    def forward(self, z, age=None, gender=None):
        out = z
        if age is not None and gender is not None:
            label = (
                Label(age, gender).to_tensor()
                if (isinstance(age, int) and isinstance(gender, int))
                else torch.cat((age, gender), 1)
            )
            out = torch.cat((out, label), 1)  # z_l
        # print(out.shape)
        out = self.fc(out)
        # print(out.shape)
        out = self._decompress(out)
        # print(out.shape)
        for i, deconv_layer in enumerate(self.deconv_layers, 1):
            out = deconv_layer(out)
            # print(out.shape)
        return out


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        self.fc = nn.Linear(in_features=in_channel, out_features=out_channel)

    def forward(self, x):
        return self.fc(x)
