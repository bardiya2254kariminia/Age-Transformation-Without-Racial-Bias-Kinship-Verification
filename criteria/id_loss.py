# import torch
# from torch import nn
# import torch.functional as F
# from torch.nn import Parameter
# import os
# import sys
# from torchvision.models import resnet34
# import math
# import consts
# from __future__ import print_function
# from __future__ import division
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Parameter
# import math


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from configs.paths_config import model_paths
# from models.encoders.model_irse import Backbone
# from torch.nn.functional import mse_loss


# class IDLoss(nn.Module):
#     def __init__(self):
#         super(IDLoss, self).__init__()
#         print("Loading ResNet ArcFace")
#         self.facenet = Backbone(
#             input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
#         )
#         self.facenet.load_state_dict(torch.load(model_paths["ir_se50"]))
#         self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
#         self.facenet.eval()
#         for module in [self.facenet, self.face_pool]:
#             for param in module.parameters():
#                 param.requires_grad = False

#     def extract_feats(self, x):
#         x = x[:, :, 35:223, 32:220]  # Crop interesting region
#         x = self.face_pool(x)
#         x_feats = self.facenet(x)
#         print(f"{x_feats.shape=}")
#         return x_feats

#     def forward(self, y_hat, y, x):
#         n_samples = x.shape[0]
#         x_feats = self.extract_feats(x)
#         y_feats = self.extract_feats(y)  # Otherwise use the feature from there
#         y_hat_feats = self.extract_feats(y_hat)
#         y_feats = y_feats.detach()
#         loss = 0
#         sim_improvement = 0
#         id_logs = []
#         count = 0
#         for i in range(n_samples):
#             diff_target = y_hat_feats[i].dot(y_feats[i])
#             diff_input = y_hat_feats[i].dot(x_feats[i])
#             diff_views = y_feats[i].dot(x_feats[i])
#             id_logs.append(
#                 {
#                     "diff_target": float(diff_target),
#                     "diff_input": float(diff_input),
#                     "diff_views": float(diff_views),
#                 }
#             )
#             loss += 1 - diff_target
#             id_diff = float(diff_target) - float(diff_views)
#             sim_improvement += id_diff
#             count += 1

#         return loss / count, sim_improvement / count, id_logs


# class Arc_Face_loss(nn.Module):
#     def __init__(
#         self,
#         backbone_resnet_type=34,
#         pretrained_backbone=True,
#         num_classes=consts.NUM_IMAGES,
#     ):
#         super(Arc_Face_loss, self).__init__()
#         self.backbone = resnet34(pretrained_backbone)
#         self.backbone.eval()
#         self.arc_face_loss = ArcMarginProduct(
#             in_features=1000, out_features=num_classes, s=30, m=0.5
#         )
#         self.criterion = nn.CrossEntropyLoss()
#         for param in self.backbone.parameters():
#             param.requires_grad = False

#     def forward(self, x, batch_number):
#         batch = x.shape[0]
#         labels = torch.zeros((batch, consts.NUM_IMAGES))
#         for i in batch:
#             labels[i, batch_number * consts.BATCH_SIZE + i] = 1
#         embeddings = self.backbone(x)
#         logits = self.arc_face_loss(embeddings, labels)
#         return self.criterion(logits, labels)


# class ArcMarginProduct(nn.Module):
#     r"""Implement of large margin arc distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin

#         cos(theta + m)
#     """

#     def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
#         super(ArcMarginProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m

#     def forward(self, input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#         # --------------------------- convert label to one-hot ---------------------------
#         # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
#         one_hot = torch.zeros(cosine.size(), device="cuda")
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + (
#             (1.0 - one_hot) * cosine
#         )  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s
#         # print(output)

#         return output


# class AddMarginProduct(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#         cos(theta) - m
#     """

#     def __init__(self, in_features, out_features, s=30.0, m=0.40):
#         super(AddMarginProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         phi = cosine - self.m
#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cosine.size(), device="cuda")
#         # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + (
#             (1.0 - one_hot) * cosine
#         )  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s
#         # print(output)

#         return output

#     def __repr__(self):
#         return (
#             self.__class__.__name__
#             + "("
#             + "in_features="
#             + str(self.in_features)
#             + ", out_features="
#             + str(self.out_features)
#             + ", s="
#             + str(self.s)
#             + ", m="
#             + str(self.m)
#             + ")"
#         )


# class SphereProduct(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         m: margin
#         cos(m*theta)
#     """

#     def __init__(self, in_features, out_features, m=4):
#         super(SphereProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.m = m
#         self.base = 1000.0
#         self.gamma = 0.12
#         self.power = 1
#         self.LambdaMin = 5.0
#         self.iter = 0
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform(self.weight)

#         # duplication formula
#         self.mlambda = [
#             lambda x: x**0,
#             lambda x: x**1,
#             lambda x: 2 * x**2 - 1,
#             lambda x: 4 * x**3 - 3 * x,
#             lambda x: 8 * x**4 - 8 * x**2 + 1,
#             lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
#         ]

#     def forward(self, input, label):
#         # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
#         self.iter += 1
#         self.lamb = max(
#             self.LambdaMin,
#             self.base * (1 + self.gamma * self.iter) ** (-1 * self.power),
#         )

#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
#         cos_theta = cos_theta.clamp(-1, 1)
#         cos_m_theta = self.mlambda[self.m](cos_theta)
#         theta = cos_theta.data.acos()
#         k = (self.m * theta / 3.14159265).floor()
#         phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
#         NormOfFeature = torch.norm(input, 2, 1)

#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cos_theta.size())
#         one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
#         one_hot.scatter_(1, label.view(-1, 1), 1)

#         # --------------------------- Calculate output ---------------------------
#         output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
#         output *= NormOfFeature.view(-1, 1)

#         return output

#     def __repr__(self):
#         return (
#             self.__class__.__name__
#             + "("
#             + "in_features="
#             + str(self.in_features)
#             + ", out_features="
#             + str(self.out_features)
#             + ", m="
#             + str(self.m)
#             + ")"
#         )


####################################################33
# import torch
# from torch import nn
# from configs.paths_config import model_paths
# from models.encoders.model_irse import Backbone


# class IDLoss(nn.Module):
#     def __init__(self):
#         super(IDLoss, self).__init__()
#         print('Loading ResNet ArcFace')
#         self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
#         self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
#         self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
#         self.facenet.eval()
#         for module in [self.facenet, self.face_pool]:
#             for param in module.parameters():
#                 param.requires_grad = False

#     def extract_feats(self, x):
#         x = x[:, :, 35:223, 32:220]  # Crop interesting region
#         x = self.face_pool(x)
#         x_feats = self.facenet(x)
#         return x_feats

#     def forward(self, y_hat, y, x):
#         n_samples = x.shape[0]
#         x_feats = self.extract_feats(x)
#         y_feats = self.extract_feats(y)  # Otherwise use the feature from there
#         y_hat_feats = self.extract_feats(y_hat)
#         y_feats = y_feats.detach()
#         loss = 0
#         sim_improvement = 0
#         id_logs = []
#         count = 0
#         for i in range(n_samples):
#             diff_target = y_hat_feats[i].dot(y_feats[i])
#             diff_input = y_hat_feats[i].dot(x_feats[i])
#             diff_views = y_feats[i].dot(x_feats[i])
#             id_logs.append({'diff_target': float(diff_target),
#                             'diff_input': float(diff_input),
#                             'diff_views': float(diff_views)})
#             loss += 1 - diff_target
#             id_diff = float(diff_target) - float(diff_views)
#             sim_improvement += id_diff
#             count += 1

#         return loss / count, sim_improvement / count, id_logs
# ######################################################
# import torch
# from torch import nn
# import os
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from configs.paths_config import model_paths
# from models.encoders.model_irse import Backbone
# from torch.nn.functional import mse_loss


# class IDLoss(nn.Module):
#     def __init__(self):
#         super(IDLoss, self).__init__()
#         print("Loading ResNet ArcFace")
#         self.facenet = Backbone(
#             input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
#         )
#         self.facenet.load_state_dict(torch.load(model_paths["ir_se50"]))
#         self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
#         self.facenet.eval()
#         for module in [self.facenet, self.face_pool]:
#             for param in module.parameters():
#                 param.requires_grad = False

#     def extract_feats(self, x):
#         x = x[:, :, 35:223, 32:220]  # Crop interesting region
#         x = self.face_pool(x)
#         x_feats = self.facenet(x)
#         print(f"{x_feats.shape=}")
#         return x_feats

#     def forward(self, y_hat, y, x):
#         n_samples = x.shape[0]
#         x_feats = self.extract_feats(x)
#         y_feats = self.extract_feats(y)  # Otherwise use the feature from there
#         y_hat_feats = self.extract_feats(y_hat)
#         y_feats = y_feats.detach()
#         loss = 0
#         sim_improvement = 0
#         id_logs = []
#         count = 0
#         for i in range(n_samples):
#             diff_target = y_hat_feats[i].dot(y_feats[i])
#             diff_input = y_hat_feats[i].dot(x_feats[i])
#             diff_views = y_feats[i].dot(x_feats[i])
#             id_logs.append(
#                 {
#                     "diff_target": float(diff_target),
#                     "diff_input": float(diff_input),
#                     "diff_views": float(diff_views),
#                 }
#             )
#             loss += 1 - diff_target
#             id_diff = float(diff_target) - float(diff_views)
#             sim_improvement += id_diff
#             count += 1

#         return loss / count, sim_improvement / count, id_logs


# class IDLOSS_ArcFace_l2(nn.Module):
#     def __init__(self):
#         super(IDLOSS_ArcFace_l2, self).__init__()
#         print("Loading ResNet ArcFace")
#         self.facenet = Backbone(
#             input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
#         )
#         self.facenet.load_state_dict(torch.load(model_paths["ir_se50"]))
#         self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
#         self.facenet.eval()
#         for module in [self.facenet, self.face_pool]:
#             for param in module.parameters():
#                 param.requires_grad = False

#     def extract_feats(self, x):
#         x = x[:, :, 35:223, 32:220]  # Crop interesting region
#         x = self.face_pool(x)
#         x_feats = self.facenet(x)
#         # print(f"{x_feats.shape=}")
#         return x_feats

#     def forward(self, x, y):
#         n_samples = x.shape[0]
#         x_feats = self.extract_feats(x)
#         y_feats = self.extract_feats(y)
#         y_feats = y_feats.detach()
#         return torch.norm(y_feats- x_feats, p=2)
from __future__ import print_function, division
import torch
from torch import nn
import torch.functional as F
from torch.nn import Parameter
import os
import sys
from torchvision.models import resnet34
import math
import consts
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Parameter
import math


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone
from torch.nn.functional import mse_loss


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print("Loading ResNet ArcFace")
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
        )
        self.facenet.load_state_dict(torch.load(model_paths["ir_se50"]))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        print(f"{x_feats.shape=}")
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append(
                {
                    "diff_target": float(diff_target),
                    "diff_input": float(diff_input),
                    "diff_views": float(diff_views),
                }
            )
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs


class Arc_Face_loss(nn.Module):
    def __init__(
        self,
        backbone_resnet_type=34,
        pretrained_backbone=True,
        num_classes=consts.NUM_IMAGES,
    ):
        super(Arc_Face_loss, self).__init__()
        self.backbone = resnet34(pretrained_backbone)
        self.backbone.eval()
        self.arc_face_loss = ArcMarginProduct(
            in_features=1000, out_features=num_classes, s=30, m=0.5
        )
        self.criterion = nn.CrossEntropyLoss()
        # self.loss_optimizer = Adam(params=self.arc_face_loss.parameters())
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x, batch_number):
        batch = x.shape[0]
        labels = []
        for i in range(batch):
            labels.append((batch_number - 1) * consts.BATCH_SIZE + i)
        labels = torch.tensor(labels).to("cuda")
        embeddings = self.backbone(x)
        logits = self.arc_face_loss(embeddings, labels)

        return self.criterion(logits, labels)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin

        cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
            (1.0 - one_hot) * cosine
        )  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device="cuda")
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
            (1.0 - one_hot) * cosine
        )  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", s="
            + str(self.s)
            + ", m="
            + str(self.m)
            + ")"
        )


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """

    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(
            self.LambdaMin,
            self.base * (1 + self.gamma * self.iter) ** (-1 * self.power),
        )

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", m="
            + str(self.m)
            + ")"
        )
