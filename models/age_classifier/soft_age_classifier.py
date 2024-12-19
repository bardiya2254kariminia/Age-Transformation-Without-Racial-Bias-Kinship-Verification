# import consts
# import torch
# from torch import nn
# class Age_Classifier(nn.Module):
#     def __init__(self):
#         super(Age_Classifier , self).__init__()
#         self.layers = nn.ModuleList()
#         self.layers.add_module(
#             "age_classifier_fc1",
#             nn.Sequential(
#                 nn.Linear(
#                     in_features=consts.NUM_Z_CHANNELS,
#                     out_features=consts.NUM_AGES
#                 )
#             )
#         )
#     def forward(self,x:torch.tensor):
#         out = x
#         for layer in self.layers:
#             out = layer(out)
#         return out


import consts
import torch
from torch import nn


class Age_Classifier(nn.Module):
    def __init__(self):
        super(Age_Classifier, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.add_module(
            "age_classifier_fc1",
            nn.Sequential(nn.Linear(in_features=512, out_features=consts.NUM_AGES)),
        )

    def forward(self, x: torch.tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
