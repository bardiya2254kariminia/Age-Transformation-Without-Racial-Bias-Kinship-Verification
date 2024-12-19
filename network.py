# # utils
# import consts

# # image  processing and logging
# from utils.pipeline_utils import *
# from tqdm import tqdm
# import logging
# import random
# from collections import OrderedDict
# import imageio
# import cv2
# from PIL import Image
# import numpy as np

# # loss and optimizers
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.functional import l1_loss, mse_loss
# from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits_loss
# from utils import common, train_utils
# from criteria import id_loss, moco_loss
# from configs import data_configs
# from criteria.lpips.lpips import LPIPS
# from torch.optim import Adam
# from torch.utils.data import DataLoader

# # modules
# from models.stylegan2.model import Generator
# from models.stylegan2.networks import Discriminator
# from models.base_modules.basic_modules import DiscriminatorImg, DiscriminatorZ, MLP
# from models.age_classifier.soft_age_classifier import Age_Classifier
# from models.encoders.psp_encoders import (
#     Encoder4Editing,
#     AgeEncoder4Editing,
#     Encoder4Editing_mini,
# )
# from models.age_modulator import resnet
# from torchvision.models import resnet18, resnet34

# # path configs
# from configs.paths_config import model_paths


# # debugging
# torch.autograd.set_detect_anomaly(True)
# from torchsummary import summary

# # ============================= MARK 3 Model ==========================================


# class Net(object):
#     def __init__(self, opts):
#         # resources
#         self.device = "cuda"
#         self.opts = opts
#         # Modules
#         self.mlp = MLP(in_channel=consts.IMAGE_LENGTH, out_channel=512)
#         self._set_encoder(pretrained=True, freeze=False, type="Encoder4Editing")
#         self.Dz = DiscriminatorZ()
#         self.Dimg = DiscriminatorImg()
#         # self.Dimg_dict = self._set_discriminators()
#         self.Age_Classifier = Age_Classifier()
#         # self.E = Encoder4Editing_mini(50, "ir_se", self.opts)

#         # creating age modulator and loadits value and train it
#         self._set_age_modulator(pretrained=True, freeze=False)

#         # creating a pretrained stylegan2 generator and freezing its value
#         # self._set_gen_stylegan(pretrained=True, freeze=True)

#         # creating our own Generator we used a pretrained stylegan 2 and train it with additional ADAI normalizations
#         self.G = Generator(
#             opts.stylegan_size,
#             consts.FAW_LENGTH,
#             8,
#             channel_multiplier=2,
#         ).to(self.device)

#         # optimizers
#         self.eg_optimizer = Adam(
#             list(self.E.parameters()) + list(self.G.parameters())
#         )  # should be optimized with each other
#         self.dz_optimizer = Adam(self.Dz.parameters())
#         self.di_optimizer = Adam(self.Dimg.parameters())
#         self.age_modulator_optimizer = Adam(self.Age_Modulator.parameters())
#         # self.di_optimizers_dict = self._set_discriminator_optimizers(
#         #     dictionary=self.Dimg_dict
#         # )
#         self.age_classifier_optimizer = Adam(self.Age_Classifier.parameters())

#         # Initialize loss
#         self.lpips_loss = LPIPS(net_type=self.opts.lpips_type).to(self.device).eval()
#         self.id_loss = id_loss.Arc_Face_loss().to(self.device).eval()
#         self.arc_face_loss_optimizer = Adam(
#             params=self.id_loss.arc_face_loss.parameters()
#         )
#         self.mse_loss = nn.MSELoss().to(self.device).eval()
#         self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device).eval()
#         # self.cpu()  # initial, can later move to cuda
#         self.cuda()  # initial, can later move to cuda

#     def _set_encoder(self, pretrained=True, freeze=False, type="Encoder4Editing"):
#         self.E = AgeEncoder4Editing(50, "ir_se", self.opts)
#         if pretrained:
#             # ckpt = torch.load(model_paths[type] , weights_only  = True)
#             # encoder_state_dict = {
#             #     key[len("encoder.") :]: value
#             #     for key, value in ckpt["state_dict"].items()
#             #     if key.startswith("encoder.")
#             # }
#             # self.E.load_state_dict(encoder_state_dict)
#             # e = AgeEncoder4Editing(50, "ir", opts).to(self.device)
#             pretrained_state_dict = torch.load(
#                 model_paths["Encoder4Editing"], weights_only=True
#             )
#             model_state_dict = self.E.state_dict()
#             for key in pretrained_state_dict.keys():
#                 if key in model_state_dict:
#                     model_state_dict[key] = pretrained_state_dict[key]
#             self.E.load_state_dict(model_state_dict)
#             for param in self.E.body.parameters():
#                 param.requires_grad = False
#             for param in self.E.input_layer.parameters():
#                 param.requires_grad = False

#                 for i in range(consts.NUM_AGES + 1):
#                     layer = self.E.styles[i]
#                     ls = None
#                     if i < 3:
#                         ls = range(2)
#                     elif i < 7:
#                         ls = range(2)
#                     else:
#                         ls = range(3)
#                     for k in ls:
#                         l = layer.convs[k * 2]
#                         for param in l.parameters():
#                             param.requires_grad = False

#         if freeze:
#             self.E.eval()
#             for param in self.E.parameters():
#                 param.requires_grad = False

#         self.E.to(self.device)

#     def _set_discriminator(self):
#         self.Dimg = Discriminator(c_dim=0, img_resolution=1024, img_channels=3)
#         ckpt = torch.load(model_paths["Discriminator_nvlab"])
#         self.Dimg.load_state_dict(ckpt, strict=True).to(self.device)

#     def _set_age_modulator(self, pretrained=True, freeze=False):
#         self.Age_Modulator = resnet.create_resnet(
#             n=consts.RESNET_TYPE,
#             lstm_size=consts.NUM_AGES,
#             emb_size=128,
#             use_pretrained=False,
#         )
#         # self.Age_Modulator = resnet.create_resnet(
#         #     n=consts.RESNET_TYPE,
#         #     lstm_size=consts.NUM_AGES,
#         #     emb_size=256,
#         #     use_pretrained = False
#         # )
#         if consts.RESNET_TYPE == 18:
#             pretrained_res = resnet18(pretrained=True).to(self.device)
#         else:
#             pretrained_res = resnet34(pretrained=True).to(self.device)

#         model_state_dict = self.Age_Modulator.state_dict()
#         pretrained_model_state_dict = pretrained_res.state_dict()

#         # deleting the model and empty the gpu  cache
#         pretrained_res.to("cpu")
#         del pretrained_res
#         torch.cuda.empty_cache()
#         # modify state_dict and loading the model
#         if pretrained:
#             for key in pretrained_model_state_dict.keys():
#                 flag = False
#                 for arg in key.split("."):
#                     for narg in ("bn", "running", "num_batches", "downsample", "fc"):
#                         if narg in arg:
#                             flag = True
#                             break
#                     if flag:
#                         break
#                 if not flag:
#                     model_state_dict[key] = pretrained_model_state_dict[key]
#             self.Age_Modulator.load_state_dict(model_state_dict)

#             for name, module in self.Age_Modulator.named_modules():
#                 if "bn" not in name:
#                     for param in module.parameters():
#                         param.requires_grad = False

#         if freeze:
#             self.Age_Modulator.eval()
#             for param in self.Age_Modulator.parameters():
#                 param.requires_grad = False
#         self.Age_Modulator.to(self.device)

#     def _set_gen_stylegan(
#         self, size=1024, style_dim=512, n_mlp=8, pretrained=False, freeze=False
#     ):
#         self.Freezed_stylegan_generator = Generator(
#             size=size, style_dim=style_dim, n_mlp=n_mlp
#         )
#         if pretrained:
#             ckpt = torch.load(
#                 model_paths["generator_style_gan2"],
#                 map_location=self.device,
#                 weights_only=True,
#             )
#             self.Freezed_stylegan_generator.load_state_dict(ckpt["g_ema"], strict=True)
#         if freeze:
#             self.Freezed_stylegan_generator.eval()
#             for param in self.Freezed_stylegan_generator.parameters():
#                 param.requires_grad = False
#         self.Freezed_stylegan_generator.to(self.device)

#     def __call__(self, *args, **kwargs):
#         self.test_single(*args, **kwargs)

#     def __repr__(self):
#         return os.linesep.join(
#             [repr(subnet) for subnet in (self.E, self.Dz, self.G)]
#         )  # used for getting information of models

#     def is_training_discriminator(self):
#         return self.opts.w_discriminator_lambda > 0

#     def get_dims_to_discriminate(self):
#         deltas_starting_dimensions = self.E.get_deltas_starting_dimensions()
#         return deltas_starting_dimensions[: self.E.progressive_stage.value + 1]

#     def is_progressive_training(self):
#         return self.opts.progressive_steps is not None

#     def calc_loss_e4e(self, x, y, y_hat, latent):
#         loss_dict = {}
#         loss = 0.0
#         id_logs = None
#         if (
#             self.opts.progressive_steps
#             and self.net.encoder.progressive_stage.value != 18
#         ):  # delta regularization loss
#             total_delta_loss = 0
#             deltas_latent_dims = self.E.get_deltas_starting_dimensions()

#             first_w = latent[:, 0, :]
#             for i in range(1, self.E.progressive_stage.value + 1):
#                 curr_dim = deltas_latent_dims[i]
#                 delta = latent[:, curr_dim, :] - first_w
#                 delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
#                 loss_dict[f"delta{i}_loss"] = float(delta_loss)
#                 total_delta_loss += delta_loss
#             loss_dict["total_delta_loss"] = float(total_delta_loss)
#             loss += self.opts.delta_norm_lambda * total_delta_loss

#         # if self.opts.id_lambda > 0:  # Similarity loss
#         #     loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
#         #     loss_dict['loss_id'] = float(loss_id)
#         #     loss_dict['id_improve'] = float(sim_improvement)
#         #     loss += loss_id * self.opts.id_lambda
#         # if self.opts.l2_lambda > 0:
#         #     loss_l2 = F.mse_loss(y_hat, y)
#         #     loss_dict['loss_l2'] = float(loss_l2)
#         #     loss += loss_l2 * self.opts.l2_lambda
#         # if self.opts.lpips_lambda > 0:
#         #     loss_lpips = self.lpips_loss(y_hat, y)
#         #     loss_dict['loss_lpips'] = float(loss_lpips)
#         #     loss += loss_lpips * self.opts.lpips_lambda
#         # loss_dict['loss'] = float(loss)
#         return loss, loss_dict, id_logs

#     def test_single(self, image_tensor, age, gender, target, watermark):

#         self.eval()
#         batch = image_tensor.repeat(consts.NUM_AGES, 1, 1, 1).to(
#             device=self.device
#         )  # N x D x H x W
#         z = self.E(batch)  # N x Z

#         gender_tensor = -torch.ones(consts.NUM_GENDERS)
#         gender_tensor[int(gender)] *= -1
#         gender_tensor = gender_tensor.repeat(
#             consts.NUM_AGES, consts.NUM_AGES // consts.NUM_GENDERS
#         )  # apply gender on all images

#         age_tensor = -torch.ones(consts.NUM_AGES, consts.NUM_AGES)
#         for i in range(consts.NUM_AGES):
#             age_tensor[i][i] *= -1  # apply the i'th age group on the i'th image

#         l = torch.cat((age_tensor, gender_tensor), 1).to(self.device)
#         z_l = torch.cat((z, l), 1)

#         generated = self.G(z_l)

#         if watermark:
#             image_tensor = image_tensor.permute(1, 2, 0)
#             image_tensor = 255 * one_sided(image_tensor.numpy())
#             image_tensor = np.ascontiguousarray(image_tensor, dtype=np.uint8)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             bottomLeftCornerOfText = (2, 25)
#             fontScale = 0.5
#             fontColor = (0, 128, 0)  # dark green, should be visible on most skin colors
#             lineType = 2
#             cv2.putText(
#                 image_tensor,
#                 "{}, {}".format(["Male", "Female"][gender], age),
#                 bottomLeftCornerOfText,
#                 font,
#                 fontScale,
#                 fontColor,
#                 lineType,
#             )
#             image_tensor = (
#                 two_sided(torch.from_numpy(image_tensor / 255.0))
#                 .float()
#                 .permute(2, 0, 1)
#             )

#         joined = torch.cat((image_tensor.unsqueeze(0), generated), 0)

#         joined = nn.ConstantPad2d(padding=4, value=-1)(joined)
#         for img_idx in (0, Label.age_transform(age) + 1):
#             for elem_idx in (0, 1, 2, 3, -4, -3, -2, -1):
#                 joined[img_idx, :, elem_idx, :] = 1  # color border white
#                 joined[img_idx, :, :, elem_idx] = 1  # color border white

#         dest = os.path.join(target, "out_{0}_{1}.png".format(gender, age))

#         # show and save the input and latest age
#         s_head_tail = False
#         if s_head_tail:
#             joined = joined[:: len(joined) - 1]  # first and last item

#         save_image_normalized(tensor=joined, filename=dest, nrow=joined.size(0))
#         print_timestamp("Saved test result to " + dest)
#         return dest

#     def calc_loss(
#         self,
#         input_images,
#         labels,
#         z,
#         generated_images,
#         generated_latents,
#         reconstruct_images,
#         reconstruct_generated_images,
#         d_z_prior,
#         d_z,
#         d_i_input,  # Dimg(input_image)
#         d_i_output,  # Dimg(generated_image)
#         batch_number,
#         is_epoch_treshold=False,
#         lambda_total_variation=0.1,
#         lambda_Dz_loss=0.45,
#         lambda_Dimg_loss=1,
#         lambda_id_loss=1,
#         lambda_age_loss=1,
#         lambda_cyclic_loss=1,
#         lambda_reconstruction_loss=1,
#     ):
#         loss = 0
#         losses = defaultdict(list)
#         # total variation loss
#         loss_total_variation = l1_loss(
#             generated_images[:, :, :, :-1], generated_images[:, :, :, 1:]
#         ) + l1_loss(generated_images[:, :, :-1, :], generated_images[:, :, 1:, :])
#         loss += (lambda_total_variation * loss_total_variation).item()
#         losses["total_variation"].append(loss_total_variation.item())

#         # Discriminator_Z_prior
#         disc_z_loss = bce_with_logits_loss(
#             d_z, torch.zeros_like(d_z)
#         ) + bce_with_logits_loss(d_z_prior, torch.ones_like(d_z_prior))
#         losses["Dz_loss"].append(disc_z_loss.item())
#         # loss += (disc_z_loss).item()

#         # ez_loss
#         ez_loss = bce_with_logits_loss(d_z, torch.ones_like(d_z))
#         losses["ez_loss"].append(ez_loss.item())
#         loss += lambda_Dz_loss * (ez_loss + disc_z_loss).item()

#         # DiscImg_loss
#         dimg_loss = bce_with_logits_loss(
#             d_i_output, torch.zeros_like(d_i_output)
#         ) + bce_with_logits_loss(d_i_input, torch.ones_like(d_i_input))
#         losses["Dimg_loss"].append(dimg_loss.item())
#         # loss += dimg_loss.item()

#         # generator_Dimg loss
#         dg_loss = bce_with_logits_loss(d_i_output, torch.ones_like(d_i_output))
#         losses["Dg_loss"].append(dg_loss.item())
#         loss += lambda_Dimg_loss * (dimg_loss + dg_loss).item()

#         # Encoder , Generator loss
#         if (
#             self.opts.progressive_steps
#             and self.net.encoder.progressive_stage.value != 18
#         ):  # delta regularization loss
#             # delta regularization loss
#             print(f"delta reg loss computed")
#             total_delta_loss = 0
#             deltas_latent_dims = self.E.get_deltas_starting_dimensions()

#             first_w = generated_latents[:, 0, :]
#             for i in range(1, self.E.progressive_stage.value + 1):
#                 curr_dim = deltas_latent_dims[i]
#                 delta = generated_latents[:, curr_dim, :] - first_w
#                 delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
#                 losses[f"delta{i}_loss"] = float(delta_loss)
#                 total_delta_loss += delta_loss
#             losses["total_delta_loss"] = float(total_delta_loss)
#             eg_loss = self.opts.delta_norm_lambda * total_delta_loss
#             losses["eg_loss"].append(eg_loss)
#             loss += eg_loss
#         else:
#             losses["eg_loss"].append(0)

#         # ID loss
#         if is_epoch_treshold:
#             id_loss = self.id_loss(input_images, batch_number)
#             losses["ID_loss"].append(id_loss)
#             loss += lambda_id_loss * id_loss
#         else:
#             id_loss = self.id_loss(generated_images, batch_number)
#             losses["ID_loss"].append(id_loss)
#             loss += lambda_id_loss * id_loss
#         # Age loss
#         age_out = self.Age_Classifier(z[:, 1:, :])
#         age_target = torch.zeros((consts.BATCH_SIZE, consts.NUM_AGES, consts.NUM_AGES))
#         for i in range(consts.NUM_AGES):
#             age_target[:, i, i] = 1
#         age_target = age_target.to(device=self.device)
#         age_loss = self.cross_entropy_loss(age_out, age_target)
#         losses["age_loss"].append(age_loss.item())
#         loss += lambda_age_loss * age_loss

#         # Cyclic loss
#         if not is_epoch_treshold:
#             loss_cyclic = l1_loss(reconstruct_images, input_images) + l1_loss(
#                 generated_images, reconstruct_generated_images
#             )
#             losses["Cyclic_loss"].append(loss_cyclic)
#             loss += lambda_cyclic_loss * loss_cyclic
#         else:
#             losses["Cyclic_loss"].append(0)

#         # Recunstruction loss
#         loss_reconstruct = l1_loss(generated_images, input_images)
#         losses["Recunstruction_loss"].append(loss_reconstruct)
#         loss += lambda_reconstruction_loss * loss_reconstruct
#         return losses, loss

#     def forward_pass(
#         self,
#         idx_to_class,
#         epoch_number,
#         batch_number,
#         images,
#         labels,
#         epoch_treshold=50,
#     ):

#         def str_to_label(text: str):
#             age, gender = text.split(".")
#             age_tensor = torch.zeros(consts.NUM_AGES)
#             gender_tensor = torch.zeros(consts.NUM_GENDERS)
#             if epoch_number < epoch_treshold:
#                 age_tensor[int(age)] = 1
#             else:
#                 age_tensor[np.random.randint(low=0, high=consts.NUM_AGES, size=1)] = 1
#             gender_tensor[int(gender)] = 1
#             return age_tensor, gender_tensor

#         def cycle_pass(images, labels, age_gender_labels=None):
#             if age_gender_labels == None:
#                 age_gender_labels = [
#                     (str_to_label(idx_to_class[l])) for l in list(labels.numpy())
#                 ]
#             age_tensor_g = [x[0] for x in age_gender_labels]
#             gender_tensor_g = [x[1] for x in age_gender_labels]
#             # print(f"{age_tensor=} ,{gender_tensor=}")
#             age_tensor_g = torch.stack(age_tensor_g, dim=0).to(self.device)
#             gender_tensor_g = torch.stack(gender_tensor_g, dim=0).to(self.device)
#             # gender_tensor = torch.tensor(gender_tensor).to(self.device)
#             # phase 1: getting style feature maps
#             z = self.E(images)
#             # only get NUM_AGES z's
#             z = z[:, : consts.NUM_AGES + 1, :]
#             # getting denoised image
#             # denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
#             #     [z[:, 0, :]],
#             #     return_latents=False,
#             #     return_features=False,
#             #     generated_image_length=256,
#             # )[0]
#             # denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
#             #     self.mlp(z[:, 0, :]), return_latents=False, return_features=False
#             # )
#             # getting f_aw
#             f_aw = self.Age_Modulator(images, age_tensor_g)  # f_aw = (batch , 512)

#             # feeding f_aw and w[0]to generator
#             z_l = torch.cat([z[:, 0, :], gender_tensor_g], dim=1)
#             reconstruct_images, latents = self.G(
#                 [z_l, f_aw], return_latents=True, generated_image_length=1024
#             )
#             # reconstruct_images, latents = self.G(
#             #     [z_l, f_aw], return_latents=True, generated_image_length=128
#             # )
#             return reconstruct_images, age_gender_labels

#         # for 50 epoch we should train the network to map the images to the same age and get reconstruction loss
#         # if epoch_number < epoch_treshold:
#         images = images.to(self.device)

#         age_gender_labels = [
#             (str_to_label(idx_to_class[l])) for l in list(labels.numpy())
#         ]
#         age_tensor = [x[0] for x in age_gender_labels]
#         gender_tensor = [x[1] for x in age_gender_labels]
#         # print(f"{age_tensor=} ,{gender_tensor=}")
#         age_tensor = torch.stack(age_tensor, dim=0).to(self.device)
#         gender_tensor = torch.stack(gender_tensor, dim=0).to(self.device)
#         # gender_tensor = torch.tensor(gender_tensor).to(self.device)
#         # phase 1: getting style feature maps
#         z = self.E(images)
#         # only get NUM_AGES z's
#         z = z[:, : consts.NUM_AGES + 1, :]
#         # getting denoised image
#         # denoised_images = (
#         #     self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
#         #         [z[:, 0, :]],
#         #         return_latents=False,
#         #         return_features=False,
#         #         generated_image_length=256,
#         #     )[0]
#         # )
#         # denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
#         #     self.mlp(z[:, 0, :]), return_latents=False, return_features=False
#         # )
#         # getting f_aw
#         # f_aw = self.Age_Modulator(denoised_images, age_tensor)  # f_aw = (batch , 512)
#         f_aw = self.Age_Modulator(images, age_tensor)  # f_aw = (batch , 512)

#         # feeding f_aw and w[0]to generator
#         z_l = torch.cat([z[:, 0, :], gender_tensor], dim=1)

#         generated, latents = self.G(
#             [z_l, f_aw], return_latents=True, generated_image_length=1024
#         )
#         # generated, latents = self.G(
#         #     [z_l, f_aw], return_latents=True, generated_image_length=128
#         # )
#         z_prior = two_sided(torch.rand_like(z[:, 0, :], device=self.device))  # [-1 : 1]

#         # min max in uniform distribution
#         d_z_prior = self.Dz(z_prior)
#         d_z = self.Dz(z[:, 0, :])
#         # minmax game in images
#         d_i_input = self.Dimg(
#             images, torch.cat([age_tensor, gender_tensor], dim=1), self.device
#         )
#         d_i_output = self.Dimg(
#             generated, torch.cat([age_tensor, gender_tensor], dim=1), self.device
#         )
#         # reconstructed image
#         if epoch_number < epoch_treshold:
#             loss_dict, loss = self.calc_loss(
#                 input_images=images,
#                 labels=labels,
#                 generated_images=generated,
#                 generated_latents=latents,
#                 reconstruct_generated_images=None,
#                 reconstruct_images=None,
#                 d_z_prior=d_z_prior,
#                 d_z=d_z,
#                 z=z,
#                 d_i_input=d_i_input,
#                 d_i_output=d_i_output,
#                 is_epoch_treshold=True,
#                 batch_number=batch_number,
#             )
#         else:
#             # calculating for the cyclic losses
#             reconstructed_images, _ = cycle_pass(images=generated, labels=labels)
#             reconstructed_generated_images, _ = cycle_pass(
#                 images=reconstructed_images,
#                 labels=labels,
#                 age_gender_labels=age_gender_labels,
#             )

#             loss_dict, loss = self.calc_loss(
#                 input_images=images,
#                 labels=labels,
#                 generated_images=generated,
#                 generated_latents=latents,
#                 reconstruct_generated_images=reconstructed_generated_images,
#                 reconstruct_images=reconstructed_images,
#                 d_z_prior=d_z_prior,
#                 d_z=d_z,
#                 z=z,
#                 d_i_input=d_i_input,
#                 d_i_output=d_i_output,
#                 is_epoch_treshold=False,
#                 batch_number=batch_number,
#             )
#         return loss_dict, loss

#     def teach(
#         self,
#         utkface_path,
#         batch_size=64,
#         epochs=1,
#         weight_decay=1e-5,
#         lr=2e-4,
#         should_plot=False,
#         betas=(0.9, 0.999),
#         valid_size=8000,
#         where_to_save=None,
#         models_saving="always",
#     ):
#         where_to_save = where_to_save or default_where_to_save()
#         dataset = get_utkface_dataset(utkface_path)
#         valid_size = valid_size or batch_size
#         valid_dataset, train_dataset = torch.utils.data.random_split(
#             dataset, (valid_size, len(dataset) - valid_size)
#         )

#         train_loader = DataLoader(
#             dataset=train_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=4,
#             drop_last=True,
#         )
#         idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

#         input_output_loss = l1_loss
#         nrow = round((2 * batch_size) ** 0.5)

#         for optimizer in (self.eg_optimizer, self.dz_optimizer, self.di_optimizer):
#             for param in ("weight_decay", "betas", "lr"):
#                 val = locals()[
#                     param
#                 ]  # search the values by the name in this scope in function
#                 if val is not None:
#                     optimizer.param_groups[0][
#                         param
#                     ] = val  # changes the optimizers value based params name
#         loss_tracker = LossTracker(plot=should_plot)
#         where_to_save_epoch = ""

#         save_count = 0
#         paths_for_gif = []

#         for epoch in range(1, epochs + 1):
#             where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch))
#             try:
#                 if not os.path.exists(where_to_save_epoch):
#                     os.makedirs(where_to_save_epoch)
#                 paths_for_gif.append(where_to_save_epoch)
#                 losses = defaultdict(lambda: [])
#                 self.train()  # move to train mode
#                 # for i, (images, labels) in enumerate(train_loader, 1):
#                 for i, (images, labels) in tqdm(
#                     enumerate(train_loader, 1), total=len(train_loader)
#                 ):

#                     # images = images.to(device=self.device)
#                     # labels = torch.stack(
#                     #     [
#                     #         str_to_tensor(idx_to_class[l], normalize=True)
#                     #         for l in list(labels.numpy())
#                     #     ]
#                     # )

#                     # labels = labels.to(device=self.device)

#                     losses, loss = self.forward_pass(
#                         idx_to_class=idx_to_class,
#                         epoch_number=epoch,
#                         batch_number=i,
#                         images=images,
#                         labels=labels,
#                         epoch_treshold=50,
#                     )
#                     self.eg_optimizer.zero_grad()
#                     self.dz_optimizer.zero_grad()
#                     self.di_optimizer.zero_grad()
#                     self.age_modulator_optimizer.zero_grad()
#                     self.arc_face_loss_optimizer.zero_grad()  # for id loss (arcface loss) training
#                     self.age_classifier_optimizer.zero_grad()

#                     # Back prop on Encoder\Generator
#                     print(f"{loss=} , {losses = }  in batch: {i}")
#                     loss.backward()

#                     self.eg_optimizer.step()
#                     self.dz_optimizer.step()
#                     self.di_optimizer.step()
#                     self.age_modulator_optimizer.step()
#                     self.arc_face_loss_optimizer.step()  # for id loss (arcface loss) training
#                     self.age_classifier_optimizer.step()

#                     now = datetime.datetime.now()

#                 logging.info(
#                     "[{h}:{m}[Epoch {e}] Loss: {t}".format(
#                         h=now.hour, m=now.minute, e=epoch, t=loss.item()
#                     )
#                 )
#                 print("----------------------------------------------")
#                 print_timestamp(f"[Epoch {epoch:d}] Loss: {losses.item():f}")
#                 to_save_models = models_saving in ("always", "tail")
#                 cp_path = self.save(where_to_save_epoch, to_save_models=to_save_models)
#                 if models_saving == "tail":
#                     prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
#                     remove_trained(prev_folder)
#                 loss_tracker.save(os.path.join(cp_path, "losses.png"))

#                 loss_tracker.append_many(**{k: mean(v) for k, v in losses.items()})
#                 loss_tracker.plot()

#                 logging.info(
#                     "[{h}:{m}[Epoch {e}] Loss: {l}".format(
#                         h=now.hour, m=now.minute, e=epoch, l=repr(loss_tracker)
#                     )
#                 )

#             except KeyboardInterrupt:
#                 print_timestamp(
#                     "{br}CTRL+C detected, saving model{br}".format(br=os.linesep)
#                 )
#                 if models_saving != "never":
#                     cp_path = self.save(where_to_save_epoch, to_save_models=True)
#                 if models_saving == "tail":
#                     prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
#                     remove_trained(prev_folder)
#                 loss_tracker.save(os.path.join(cp_path, "losses.png"))
#                 raise

#         if models_saving == "last":
#             cp_path = self.save(where_to_save_epoch, to_save_models=True)
#         loss_tracker.plot()

#     def _mass_fn(self, fn_name, *args, **kwargs):
#         """Apply a function to all possible Net's components.

#         :return:
#         """
#         for class_attr in dir(self):
#             if not class_attr.startswith("_"):
#                 class_attr_value = getattr(self, class_attr)
#                 # print(f"Class attribute: {class_attr}, Type: {type(class_attr_obj)}")
#                 if hasattr(class_attr_value, fn_name) and callable(class_attr_value):
#                     fn = getattr(class_attr_value, fn_name)
#                     if callable(fn):
#                         fn(*args, **kwargs)

#     def to(self, device):
#         self._mass_fn("to", device=device)

#     def cpu(self):
#         self._mass_fn("cpu")
#         self.device = torch.device("cpu")

#     def cuda(self):
#         self._mass_fn("cuda")
#         self.device = torch.device("cuda")

#     def eval(self):
#         """Move Net to evaluation mode.

#         :return:
#         """
#         self._mass_fn("eval")

#     def train(self):
#         """Move Net to training mode.

#         :return:
#         """
#         self._mass_fn("train")

#     def save(self, path, to_save_models=True):
#         """Save all state dicts of Net's components.

#         :return:
#         """
#         if not os.path.isdir(path):
#             os.mkdir(path)

#         saved = []
#         # if to_save_models:
#         #     for class_attr_name in dir(self):
#         #         if not class_attr_name.startswith("_"):
#         #             class_attr = getattr(self, class_attr_name)
#         #             if hasattr(class_attr, "state_dict"):
#         #                 state_dict = class_attr.state_dict()
#         #                 fname = os.path.join(
#         #                     path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name)
#         #                 )
#         #                 torch.save(state_dict, fname)
#         #                 saved.append(class_attr_name)
#         models = [self.E, self.Age_Modulator, self.G]
#         models = {
#             "AgeEncoder4Editing": self.E,
#             "Age_Modulator": self.Age_Modulator,
#             "Generator_Stylegan2": self.G,
#         }
#         for k, v in models.items():
#             state_dict = v.state_dict()
#             torch.save(state_dict, os.path.join(path, k) + ".pt")
#             saved.append(v)
#         if saved:  # if it's not None
#             print_timestamp("Saved {} to {}".format(", ".join(saved), path))
#         elif to_save_models:
#             raise FileNotFoundError("Nothing was saved to {}".format(path))
#         return path

#     def load(self, path, slim=True):
#         """Load all state dicts of Net's components.

#         :return:
#         """
#         loaded = []
#         for class_attr_name in dir(self):
#             if (not class_attr_name.startswith("_")) and (
#                 (not slim) or (class_attr_name in ("E", "G"))
#             ):
#                 class_attr = getattr(self, class_attr_name)
#                 fname = os.path.join(
#                     path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name)
#                 )
#                 if hasattr(class_attr, "load_state_dict") and os.path.exists(fname):
#                     class_attr.load_state_dict(torch.load(fname)())
#                     loaded.append(class_attr_name)
#         if loaded:
#             print_timestamp("Loaded {} from {}".format(", ".join(loaded), path))
#         else:
#             raise FileNotFoundError("Nothing was loaded from {}".format(path))


# utils
import consts
from datasets.datasets import Celeba_10000_dataset

# image  processing and logging
from utils.pipeline_utils import *
from tqdm import tqdm
import logging
import random
from collections import OrderedDict
import imageio

# import cv2
from PIL import Image
import numpy as np

# loss and optimizers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits_loss
from utils import common, train_utils
from criteria import id_loss, moco_loss
from configs import data_configs
from criteria.lpips.lpips import LPIPS
from torch.optim import Adam
from torch.utils.data import DataLoader

# modules
from models.stylegan2.model import Generator
from models.base_modules.basic_modules import DiscriminatorImg, DiscriminatorZ, MLP
from models.age_classifier.soft_age_classifier import Age_Classifier
from models.encoders.psp_encoders import (
    Encoder4Editing,
    AgeEncoder4Editing,
    Encoder4Editing_mini,
)
from models.age_modulator import resnet
from torchvision.models import resnet18, resnet34

# path configs
from configs.paths_config import model_paths


# debugging
torch.autograd.set_detect_anomaly(True)
from torchsummary import summary

# ============================= MARK 3 Model ==========================================


class Net(object):
    def __init__(self, opts):
        # resources
        self.device = "cuda"
        self.opts = opts
        # Modules
        self.mlp = MLP(in_channel=consts.IMAGE_LENGTH, out_channel=512)
        self._set_encoder(pretrained=True, freeze=False, type="Encoder4Editing")
        self.Dz = DiscriminatorZ()
        self.Dimg = DiscriminatorImg()
        # self.Dimg_dict = self._set_discriminators()
        self.Age_Classifier = Age_Classifier()
        # self.E = Encoder4Editing_mini(50, "ir_se", self.opts)

        # creating age modulator and loadits value and train it
        self._set_age_modulator(pretrained=True, freeze=False)

        # creating a pretrained stylegan2 generator and freezing its value
        self._set_gen_stylegan(pretrained=True, freeze=True)

        # creating our own Generator we used a pretrained stylegan 2 and train it with additional ADAI normalizations
        self.G = Generator(
            opts.stylegan_size,
            consts.FAW_LENGTH,
            8,
            channel_multiplier=2,
        )

        # optimizers
        self.eg_optimizer = Adam(
            list(self.E.parameters()) + list(self.G.parameters())
        )  # should be optimized with each other
        self.dz_optimizer = Adam(self.Dz.parameters())
        self.di_optimizer = Adam(self.Dimg.parameters())
        self.age_modulator_optimizer = Adam(self.Age_Modulator.parameters())
        # self.di_optimizers_dict = self._set_discriminator_optimizers(
        #     dictionary=self.Dimg_dict
        # )
        self.age_classifier_optimizer = Adam(self.Age_Classifier.parameters())

        # Initialize loss
        self.lpips_loss = LPIPS(net_type=self.opts.lpips_type).to(self.device).eval()
        self.id_loss = id_loss.Arc_Face_loss().to(self.device).eval()
        self.arc_face_loss_optimizer = Adam(
            params=self.id_loss.arc_face_loss.parameters()
        )
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device).eval()
        # self.cpu()  # initial, can later move to cuda
        self.cuda()  # initial, can later move to cuda

    def _set_encoder(self, pretrained=True, freeze=False, type="Encoder4Editing"):
        self.E = AgeEncoder4Editing(50, "ir_se", self.opts).to(self.device)
        if pretrained:
            # ckpt = torch.load(model_paths[type] , weights_only  = True)
            # encoder_state_dict = {
            #     key[len("encoder.") :]: value
            #     for key, value in ckpt["state_dict"].items()
            #     if key.startswith("encoder.")
            # }
            # self.E.load_state_dict(encoder_state_dict)
            # e = AgeEncoder4Editing(50, "ir", opts).to(self.device)
            pretrained_state_dict = torch.load(
                model_paths["Encoder4Editing"], weights_only=True
            )
            model_state_dict = self.E.state_dict()
            for key in pretrained_state_dict.keys():
                if key in model_state_dict:
                    model_state_dict[key] = pretrained_state_dict[key]
            self.E.load_state_dict(model_state_dict)
            for param in self.E.body.parameters():
                param.requires_grad = False
            for param in self.E.input_layer.parameters():
                param.requires_grad = False

            #   for i in range(consts.NUM_AGES+1):
            #     layer = self.E.styles[i]
            #     ls = None
            #     if i < 3:
            #       ls = range(2)
            #     elif i < 7:
            #       ls = range(2)
            #     else:
            #       ls = range(3)
            #     for k in ls:
            #       l = layer.convs[k*2]
            #       for  param in l.parameters():
            #         param.requires_grad = False

        if freeze:
            self.E.eval()
            for param in self.E.parameters():
                param.requires_grad = False

    def _set_discriminators(self):
        num_age_classes = consts.NUM_AGES
        dimg_dict = {}
        for i in range(num_age_classes):
            dimg_dict[i] = DiscriminatorImg()
        return dimg_dict

    def _set_discriminator_optimizers(self, dictionary):
        dimg_optimizers = {}
        for i in range(consts.NUM_AGES):
            dimg_optimizers[i] = Adam(
                params=dictionary[i].parameters,
                lr=self.opts.lr,
                weight_decay=self.opts.weight_decay,
                betas=(self.opts.b1, self.opts.b2),
            )
        return dimg_optimizers

    def _set_age_modulator(self, pretrained=True, freeze=False):
        self.Age_Modulator = resnet.create_resnet(
            n=consts.RESNET_TYPE,
            # lstm_size=consts.NUM_AGES,
            lstm_size=100,
            emb_size=256,
            use_pretrained=False,
        )
        if consts.RESNET_TYPE == 18:
            pretrained_res = resnet18(pretrained=True).to(self.device)
        else:
            pretrained_res = resnet34(pretrained=True).to(self.device)

        model_state_dict = self.Age_Modulator.state_dict()
        pretrained_model_state_dict = pretrained_res.state_dict()

        # deleting the model and empty the gpu  cache
        pretrained_res.to("cpu")
        del pretrained_res
        torch.cuda.empty_cache()
        # modify state_dict and loading the model
        if pretrained:
            for key in pretrained_model_state_dict.keys():
                flag = False
                for arg in key.split("."):
                    for narg in ("bn", "running", "num_batches", "downsample", "fc"):
                        if narg in arg:
                            flag = True
                            break
                    if flag:
                        break
                if not flag:
                    model_state_dict[key] = pretrained_model_state_dict[key]
            self.Age_Modulator.load_state_dict(model_state_dict)

            for name, module in self.Age_Modulator.named_modules():
                if "bn" not in name:
                    for param in module.parameters():
                        param.requires_grad = False

        if freeze:
            self.Age_Modulator.eval()
            for param in self.Age_Modulator.parameters():
                param.requires_grad = False

    def _set_gen_stylegan(
        self, size=1024, style_dim=512, n_mlp=8, pretrained=False, freeze=False
    ):
        self.Freezed_stylegan_generator = Generator(
            size=size, style_dim=style_dim, n_mlp=n_mlp
        )
        if pretrained:
            ckpt = torch.load(
                model_paths["generator_style_gan2"],
                map_location=self.device,
                weights_only=True,
            )
            self.Freezed_stylegan_generator.load_state_dict(ckpt["g_ema"], strict=True)
        if freeze:
            self.Freezed_stylegan_generator.eval()
            for param in self.Freezed_stylegan_generator.parameters():
                param.requires_grad = False

    def __call__(self, *args, **kwargs):
        self.test_single(*args, **kwargs)

    def __repr__(self):
        return os.linesep.join(
            [repr(subnet) for subnet in (self.E, self.Dz, self.G)]
        )  # used for getting information of models

    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0

    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.E.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[: self.E.progressive_stage.value + 1]

    def is_progressive_training(self):
        return self.opts.progressive_steps is not None

    def calc_loss_e4e(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if (
            self.opts.progressive_steps
            and self.net.encoder.progressive_stage.value != 18
        ):  # delta regularization loss
            total_delta_loss = 0
            deltas_latent_dims = self.E.get_deltas_starting_dimensions()

            first_w = latent[:, 0, :]
            for i in range(1, self.E.progressive_stage.value + 1):
                curr_dim = deltas_latent_dims[i]
                delta = latent[:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
                loss_dict[f"delta{i}_loss"] = float(delta_loss)
                total_delta_loss += delta_loss
            loss_dict["total_delta_loss"] = float(total_delta_loss)
            loss += self.opts.delta_norm_lambda * total_delta_loss

        # if self.opts.id_lambda > 0:  # Similarity loss
        #     loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
        #     loss_dict['loss_id'] = float(loss_id)
        #     loss_dict['id_improve'] = float(sim_improvement)
        #     loss += loss_id * self.opts.id_lambda
        # if self.opts.l2_lambda > 0:
        #     loss_l2 = F.mse_loss(y_hat, y)
        #     loss_dict['loss_l2'] = float(loss_l2)
        #     loss += loss_l2 * self.opts.l2_lambda
        # if self.opts.lpips_lambda > 0:
        #     loss_lpips = self.lpips_loss(y_hat, y)
        #     loss_dict['loss_lpips'] = float(loss_lpips)
        #     loss += loss_lpips * self.opts.lpips_lambda
        # loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def test_single(self, image_tensor, age, gender, target, watermark):

        self.eval()
        batch = image_tensor.repeat(consts.NUM_AGES, 1, 1, 1).to(
            device=self.device
        )  # N x D x H x W
        z = self.E(batch)  # N x Z

        gender_tensor = -torch.ones(consts.NUM_GENDERS)
        gender_tensor[int(gender)] *= -1
        gender_tensor = gender_tensor.repeat(
            consts.NUM_AGES, consts.NUM_AGES // consts.NUM_GENDERS
        )  # apply gender on all images

        age_tensor = -torch.ones(consts.NUM_AGES, consts.NUM_AGES)
        for i in range(consts.NUM_AGES):
            age_tensor[i][i] *= -1  # apply the i'th age group on the i'th image

        l = torch.cat((age_tensor, gender_tensor), 1).to(self.device)
        z_l = torch.cat((z, l), 1)

        generated = self.G(z_l)

        if watermark:
            image_tensor = image_tensor.permute(1, 2, 0)
            image_tensor = 255 * one_sided(image_tensor.numpy())
            image_tensor = np.ascontiguousarray(image_tensor, dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (2, 25)
            fontScale = 0.5
            fontColor = (0, 128, 0)  # dark green, should be visible on most skin colors
            lineType = 2
            cv2.putText(
                image_tensor,
                "{}, {}".format(["Male", "Female"][gender], age),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )
            image_tensor = (
                two_sided(torch.from_numpy(image_tensor / 255.0))
                .float()
                .permute(2, 0, 1)
            )

        joined = torch.cat((image_tensor.unsqueeze(0), generated), 0)

        joined = nn.ConstantPad2d(padding=4, value=-1)(joined)
        for img_idx in (0, Label.age_transform(age) + 1):
            for elem_idx in (0, 1, 2, 3, -4, -3, -2, -1):
                joined[img_idx, :, elem_idx, :] = 1  # color border white
                joined[img_idx, :, :, elem_idx] = 1  # color border white

        dest = os.path.join(target, "out_{0}_{1}.png".format(gender, age))

        # show and save the input and latest age
        s_head_tail = False
        if s_head_tail:
            joined = joined[:: len(joined) - 1]  # first and last item

        save_image_normalized(tensor=joined, filename=dest, nrow=joined.size(0))
        print_timestamp("Saved test result to " + dest)
        return dest

    def calc_loss(
        self,
        input_images,
        labels,
        z,
        generated_images,
        generated_latents,
        reconstruct_images,
        reconstruct_generated_images,
        d_z_prior,
        d_z,
        d_i_input,  # Dimg(input_image)
        d_i_output,  # Dimg(generated_image)
        batch_number,
        is_epoch_treshold=False,
        lambda_total_variation=0.1,
        lambda_Dz_loss=0.45,
        lambda_Dimg_loss=1,
        lambda_id_loss=1,
        lambda_age_loss=1,
        lambda_cyclic_loss=1,
        lambda_reconstruction_loss=1,
    ):
        loss = 0
        losses = defaultdict(list)
        # total variation loss
        loss_total_variation = l1_loss(
            generated_images[:, :, :, :-1], generated_images[:, :, :, 1:]
        ) + l1_loss(generated_images[:, :, :-1, :], generated_images[:, :, 1:, :])
        loss += (lambda_total_variation * loss_total_variation).item()
        losses["total_variation"].append(loss_total_variation.item())

        # Discriminator_Z_prior
        disc_z_loss = bce_with_logits_loss(
            d_z, torch.zeros_like(d_z)
        ) + bce_with_logits_loss(d_z_prior, torch.ones_like(d_z_prior))
        losses["Dz_loss"].append(disc_z_loss.item())
        # loss += (disc_z_loss).item()

        # ez_loss
        ez_loss = bce_with_logits_loss(d_z, torch.ones_like(d_z))
        losses["ez_loss"].append(ez_loss.item())
        loss += lambda_Dz_loss * (ez_loss + disc_z_loss).item()

        # DiscImg_loss
        dimg_loss = bce_with_logits_loss(
            d_i_output, torch.zeros_like(d_i_output)
        ) + bce_with_logits_loss(d_i_input, torch.ones_like(d_i_input))
        losses["Dimg_loss"].append(dimg_loss.item())
        # loss += dimg_loss.item()

        # generator_Dimg loss
        dg_loss = bce_with_logits_loss(d_i_output, torch.ones_like(d_i_output))
        losses["Dg_loss"].append(dg_loss.item())
        loss += lambda_Dimg_loss * (dimg_loss + dg_loss).item()

        # Encoder , Generator loss
        if (
            self.opts.progressive_steps
            and self.net.encoder.progressive_stage.value != 18
        ):  # delta regularization loss
            # delta regularization loss
            print(f"delta reg loss computed")
            total_delta_loss = 0
            deltas_latent_dims = self.E.get_deltas_starting_dimensions()

            first_w = generated_latents[:, 0, :]
            for i in range(1, self.E.progressive_stage.value + 1):
                curr_dim = deltas_latent_dims[i]
                delta = generated_latents[:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
                losses[f"delta{i}_loss"] = float(delta_loss)
                total_delta_loss += delta_loss
            losses["total_delta_loss"] = float(total_delta_loss)
            eg_loss = self.opts.delta_norm_lambda * total_delta_loss
            losses["eg_loss"].append(eg_loss)
            loss += eg_loss
        else:
            losses["eg_loss"].append(0)

        # ID loss
        if is_epoch_treshold:
            id_loss = self.id_loss(input_images, batch_number)
            losses["ID_loss"].append(id_loss)
            loss += lambda_id_loss * id_loss
        else:
            id_loss = self.id_loss(generated_images, batch_number)
            losses["ID_loss"].append(id_loss)
            loss += lambda_id_loss * id_loss
        # Age loss
        age_out = self.Age_Classifier(z[:, 1:, :])
        age_target = torch.zeros((consts.BATCH_SIZE, consts.NUM_AGES, consts.NUM_AGES))
        for i in range(consts.NUM_AGES):
            age_target[:, i, i] = 1
        age_target = age_target.to(device=self.device)
        age_loss = self.cross_entropy_loss(age_out, age_target)
        losses["age_loss"].append(age_loss.item())
        loss += lambda_age_loss * age_loss

        # Cyclic loss
        if not is_epoch_treshold:
            loss_cyclic = l1_loss(reconstruct_images, input_images) + l1_loss(
                generated_images, reconstruct_generated_images
            )
            losses["Cyclic_loss"].append(loss_cyclic)
            loss += lambda_cyclic_loss * loss_cyclic
        else:
            losses["Cyclic_loss"].append(0)

        # Recunstruction loss
        loss_reconstruct = l1_loss(generated_images, input_images)
        losses["Recunstruction_loss"].append(loss_reconstruct)
        loss += lambda_reconstruction_loss * loss_reconstruct
        return losses, loss


    def calc_loss2(
        self,
        input_images,
        # labels,
        z,
        generated_images,
        generated_latents,
        reconstruct_images,
        reconstruct_generated_images,
        d_z_prior,
        d_z,
        d_i_input,  # Dimg(input_image)
        d_i_output,  # Dimg(generated_image)
        batch_number,
        is_epoch_treshold=False,
        lambda_total_variation=0.1,
        lambda_Dz_loss=0.45,
        lambda_Dimg_loss=1,
        lambda_id_loss=1,
        lambda_age_loss=1,
        lambda_cyclic_loss=1,
        lambda_reconstruction_loss=1,
    ):
        loss = 0
        losses = defaultdict(list)
        # total variation loss
        loss_total_variation = l1_loss(
            generated_images[:, :, :, :-1], generated_images[:, :, :, 1:]
        ) + l1_loss(generated_images[:, :, :-1, :], generated_images[:, :, 1:, :])
        loss += (lambda_total_variation * loss_total_variation).item()
        losses["total_variation"].append(loss_total_variation.item())

        # Discriminator_Z_prior
        disc_z_loss = bce_with_logits_loss(
            d_z, torch.zeros_like(d_z)
        ) + bce_with_logits_loss(d_z_prior, torch.ones_like(d_z_prior))
        losses["Dz_loss"].append(disc_z_loss.item())
        # loss += (disc_z_loss).item()

        # ez_loss
        ez_loss = bce_with_logits_loss(d_z, torch.ones_like(d_z))
        losses["ez_loss"].append(ez_loss.item())
        loss += lambda_Dz_loss * (ez_loss + disc_z_loss).item()

        # DiscImg_loss
        dimg_loss = bce_with_logits_loss(
            d_i_output, torch.zeros_like(d_i_output)
        ) + bce_with_logits_loss(d_i_input, torch.ones_like(d_i_input))
        losses["Dimg_loss"].append(dimg_loss.item())
        # loss += dimg_loss.item()

        # generator_Dimg loss
        dg_loss = bce_with_logits_loss(d_i_output, torch.ones_like(d_i_output))
        losses["Dg_loss"].append(dg_loss.item())
        loss += lambda_Dimg_loss * (dimg_loss + dg_loss).item()

        # Encoder , Generator loss
        if (
            self.opts.progressive_steps
            and self.net.encoder.progressive_stage.value != 18
        ):  # delta regularization loss
            # delta regularization loss
            print(f"delta reg loss computed")
            total_delta_loss = 0
            deltas_latent_dims = self.E.get_deltas_starting_dimensions()

            first_w = generated_latents[:, 0, :]
            for i in range(1, self.E.progressive_stage.value + 1):
                curr_dim = deltas_latent_dims[i]
                delta = generated_latents[:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
                losses[f"delta{i}_loss"] = float(delta_loss)
                total_delta_loss += delta_loss
            losses["total_delta_loss"] = float(total_delta_loss)
            eg_loss = self.opts.delta_norm_lambda * total_delta_loss
            losses["eg_loss"].append(eg_loss)
            loss += eg_loss
        else:
            losses["eg_loss"].append(0)

        # ID loss
        if is_epoch_treshold:
            id_loss = self.id_loss(input_images, batch_number)
            losses["ID_loss"].append(id_loss)
            loss += lambda_id_loss * id_loss
        else:
            id_loss = self.id_loss(generated_images, batch_number)
            losses["ID_loss"].append(id_loss)
            loss += lambda_id_loss * id_loss
        # Age loss
        age_out = self.Age_Classifier(z[:, 1:, :])
        age_target = torch.zeros((consts.BATCH_SIZE, consts.NUM_AGES, consts.NUM_AGES))
        for i in range(consts.NUM_AGES):
            age_target[:, i, i] = 1
        age_target = age_target.to(device=self.device)
        age_loss = self.cross_entropy_loss(age_out, age_target)
        losses["age_loss"].append(age_loss.item())
        loss += lambda_age_loss * age_loss

        # Cyclic loss
        if not is_epoch_treshold:
            loss_cyclic = l1_loss(reconstruct_images, input_images) + l1_loss(
                generated_images, reconstruct_generated_images
            )
            losses["Cyclic_loss"].append(loss_cyclic)
            loss += lambda_cyclic_loss * loss_cyclic
        else:
            losses["Cyclic_loss"].append(0)

        # Recunstruction loss
        loss_reconstruct = l1_loss(generated_images, input_images)
        losses["Recunstruction_loss"].append(loss_reconstruct)
        loss += lambda_reconstruction_loss * loss_reconstruct
        return losses, loss


    def forward_pass(
        self,
        idx_to_class,
        epoch_number,
        batch_number,
        images,
        labels,
        epoch_treshold=50,
    ):

        def str_to_label(text: str):
            age, gender = text.split(".")
            age_tensor = torch.zeros(consts.NUM_AGES)
            gender_tensor = torch.zeros(consts.NUM_GENDERS)
            if epoch_number < epoch_treshold:
                age_tensor[int(age)] = 1
            else:
                age_tensor[np.random.randint(low=0, high=consts.NUM_AGES, size=1)] = 1
            gender_tensor[int(gender)] = 1
            return age_tensor, gender_tensor

        def cycle_pass(images, labels, age_gender_labels=None):
            if age_gender_labels == None:
                age_gender_labels = [
                    (str_to_label(idx_to_class[l])) for l in list(labels.numpy())
                ]
            age_tensor_g = [x[0] for x in age_gender_labels]
            gender_tensor_g = [x[1] for x in age_gender_labels]
            # print(f"{age_tensor=} ,{gender_tensor=}")
            age_tensor_g = torch.stack(age_tensor_g, dim=0).to(self.device)
            gender_tensor_g = torch.stack(gender_tensor_g, dim=0).to(self.device)
            # gender_tensor = torch.tensor(gender_tensor).to(self.device)
            # phase 1: getting style feature maps
            z = self.E(images)
            # only get NUM_AGES z's
            gender_tensor_expanded = gender_tensor.unsqueeze(1).repeat(
                1, consts.NUM_AGES + 1, 1
            )
            z_l = torch.cat([z, gender_tensor_expanded], dim=2)
            z = z[:, : consts.NUM_AGES + 1, :]
            # getting denoised image
            denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
                [z[:, 0, :]],
                return_latents=False,
                return_features=False,
                generated_image_length=256,
            )[
                0
            ]
            # denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
            #     self.mlp(z[:, 0, :]), return_latents=False, return_features=False
            # )
            # getting f_aw
            f_aw = self.Age_Modulator(
                denoised_images, age_tensor_g
            )  # f_aw = (batch , 512)
            f_aw = torch.unsqueeze(f_aw, dim=1)
            g_input = torch.cat([z_l, f_aw], dim=1)
            # feeding f_aw and w[0]to generator
            generated, latents = self.G(
                [g_input], return_latents=True, generated_image_length=128
            )
            return generated, age_gender_labels

        # for 50 epoch we should train the network to map the images to the same age and get reconstruction loss
        # if epoch_number < epoch_treshold:
        images = images.to(self.device)

        age_gender_labels = [
            (str_to_label(idx_to_class[l])) for l in list(labels.numpy())
        ]
        age_tensor = [x[0] for x in age_gender_labels]
        gender_tensor = [x[1] for x in age_gender_labels]
        # print(f"{age_tensor=} ,{gender_tensor=}")
        age_tensor = torch.stack(age_tensor, dim=0).to(self.device)
        gender_tensor = torch.stack(gender_tensor, dim=0).to(self.device)
        # gender_tensor = torch.tensor(gender_tensor).to(self.device)
        # phase 1: getting style feature maps
        z = self.E(images)
        # only get NUM_AGES z's
        gender_tensor_expanded = gender_tensor.unsqueeze(1).repeat(
            1, consts.NUM_AGES + 1, 1
        )
        z_l = torch.cat([z, gender_tensor_expanded], dim=2)
        z = z[:, : consts.NUM_AGES + 1, :]
        # getting denoised image
        denoised_images = (
            self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
                [z[:, 0, :]],
                return_latents=False,
                return_features=False,
                generated_image_length=256,
            )[0]
        )
        # denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
        #     self.mlp(z[:, 0, :]), return_latents=False, return_features=False
        # )
        # getting f_aw
        f_aw = self.Age_Modulator(denoised_images, age_tensor)  # f_aw = (batch , 512)
        f_aw = torch.unsqueeze(f_aw, dim=1)
        g_input = torch.cat([z_l, f_aw], dim=1)

        # feeding f_aw and w[0]to generator
        generated, latents = self.G(
            [g_input], return_latents=True, generated_image_length=128
        )
        z_prior = two_sided(torch.rand_like(z[:, 0, :], device=self.device))  # [-1 : 1]

        # min max in uniform distribution
        d_z_prior = self.Dz(z_prior)
        d_z = self.Dz(z[:, 0, :])
        # minmax game in images
        d_i_input = self.Dimg(
            images, torch.cat([age_tensor, gender_tensor], dim=1), self.device
        )
        d_i_output = self.Dimg(
            generated, torch.cat([age_tensor, gender_tensor], dim=1), self.device
        )
        # reconstructed image
        if epoch_number < epoch_treshold:
            loss_dict, loss = self.calc_loss(
                input_images=images,
                labels=labels,
                generated_images=generated,
                generated_latents=latents,
                reconstruct_generated_images=None,
                reconstruct_images=None,
                d_z_prior=d_z_prior,
                d_z=d_z,
                z=z,
                d_i_input=d_i_input,
                d_i_output=d_i_output,
                is_epoch_treshold=True,
                batch_number=batch_number,
            )
        else:
            # calculating for the cyclic losses
            reconstructed_images, _ = cycle_pass(images=generated, labels=labels)
            reconstructed_generated_images, _ = cycle_pass(
                images=reconstructed_images,
                labels=labels,
                age_gender_labels=age_gender_labels,
            )

            loss_dict, loss = self.calc_loss(
                input_images=images,
                labels=labels,
                generated_images=generated,
                generated_latents=latents,
                reconstruct_generated_images=reconstructed_generated_images,
                reconstruct_images=reconstructed_images,
                d_z_prior=d_z_prior,
                d_z=d_z,
                z=z,
                d_i_input=d_i_input,
                d_i_output=d_i_output,
                is_epoch_treshold=False,
                batch_number=batch_number,
            )
        return loss_dict, loss

    # def forward_pass2(
    #     self,
    #     epoch_number,
    #     batch_number,
    #     images,
    #     ages,
    #     genders,
    #     epoch_treshold=50,
    # ):

    #     def str_to_label(text: str):
    #         age, gender = text.split(".")
    #         age_tensor = torch.zeros(consts.NUM_AGES)
    #         gender_tensor = torch.zeros(consts.NUM_GENDERS)
    #         if epoch_number < epoch_treshold:
    #             age_tensor[int(age)] = 1
    #         else:
    #             age_tensor[np.random.randint(low=0, high=consts.NUM_AGES, size=1)] = 1
    #         gender_tensor[int(gender)] = 1
    #         return age_tensor, gender_tensor

    #     def cycle_pass(images, base_ages, genders , change_age):

    #         # if age_gender_labels == None:
    #         #     age_gender_labels = [
    #         #         (str_to_label(idx_to_class[l])) for l in list(labels.numpy())
    #         #     ]
    #         age_tensor_g = base_ages
    #         if change_age:
    #             age_tensor_g = torch.randint(10, 70, base_ages.shape)
    #         gender_tensor_g = genders
    #         # phase 1: getting style feature maps
    #         z = self.E(images)
    #         # only get NUM_AGES z's
    #         gender_tensor_expanded = gender_tensor.unsqueeze(1).repeat(
    #             1, consts.NUM_AGES + 1, 1
    #         )
    #         z_l = torch.cat([z, gender_tensor_expanded], dim=2)
    #         z = z[:, : consts.NUM_AGES + 1, :]
    #         # getting denoised image
    #         denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
    #             [z[:, 0, :]],
    #             return_latents=False,
    #             return_features=False,
    #             generated_image_length=256,
    #         )[0]
    #         # getting f_aw
    #         f_aw = self.Age_Modulator(
    #             denoised_images, age_tensor_g
    #         )  # f_aw = (batch , 512)
    #         f_aw = torch.unsqueeze(f_aw, dim=1)
    #         g_input = torch.cat([z_l, f_aw], dim=1)
    #         # feeding f_aw and w[0]to generator
    #         generated, latents = self.G(
    #             [g_input], return_latents=True, generated_image_length=128
    #         )
    #         return generated, gender_tensor_g , age_tensor_g

    #     # for 50 epoch we should train the network to map the images to the same age and get reconstruction loss
    #     # if epoch_number < epoch_treshold:
    #     images = images.to(self.device)
    #     base_age_tensor = ages.to(self.device)
    #     gender_tensor = genders.to(self.device)
    #     # if epoch_number < epoch_treshold:
    #     # target_age_tensor = torch.randint(10, 70, (consts.BATCH_SIZE, 1))
    #     age_tensor = base_age_tensor
    #     # phase 1: getting style feature maps
    #     z = self.E(images)
    #     # only get NUM_AGES z's
    #     gender_tensor_expanded = gender_tensor.unsqueeze(1).repeat(
    #         1, consts.NUM_AGES + 1, 1
    #     )
    #     z_l = torch.cat([z, gender_tensor_expanded], dim=2)
    #     z = z[:, : consts.NUM_AGES + 1, :]
    #     # getting denoised image
    #     denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
    #         [z[:, 0, :]],
    #         return_latents=False,
    #         return_features=False,
    #         generated_image_length=256,
    #     )[
    #         0
    #     ]
    #     # denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
    #     #     self.mlp(z[:, 0, :]), return_latents=False, return_features=False
    #     # )
    #     # getting f_aw
    #     f_aw = self.Age_Modulator(
    #         denoised_images, age_tensor
    #     )  # f_aw = (batch , 512)
    #     f_aw = torch.unsqueeze(f_aw, dim=1)
    #     g_input = torch.cat([z_l, f_aw], dim=1)

    #     # feeding f_aw and w[0]to generator
    #     generated, latents = self.G(
    #         [g_input], return_latents=True, generated_image_length=128
    #     )
    #     z_prior = two_sided(
    #         torch.rand_like(z[:, 0, :], device=self.device)
    #     )  # [-1 : 1]

    #     # min max in uniform distribution
    #     d_z_prior = self.Dz(z_prior)
    #     d_z = self.Dz(z[:, 0, :])
    #     # minmax game in images
    #     d_i_input = self.Dimg(
    #         images, torch.cat([age_tensor, gender_tensor], dim=1), self.device
    #     )
    #     d_i_output = self.Dimg(
    #         generated, torch.cat([age_tensor, gender_tensor], dim=1), self.device
    #     )
    #     if epoch_number < epoch_treshold:
    #         # reconstructed image
    #         loss_dict, loss = self.calc_loss2(
    #             input_images=images,
    #             generated_images=generated,
    #             generated_latents=latents,
    #             reconstruct_generated_images=None,
    #             reconstruct_images=None,
    #             d_z_prior=d_z_prior,
    #             d_z=d_z,
    #             z=z,
    #             d_i_input=d_i_input,
    #             d_i_output=d_i_output,
    #             is_epoch_treshold=True,
    #             batch_number=batch_number,
    #         )
    #     else:
    #         # calculating for the cyclic losses
    #         reconstructed_images, gender_tensor_reconstruct , age_tensor_reconstruct = cycle_pass(images=generated, base_ages=base_age_tensor , genders=gender_tensor ,change_age=True)
    #         reconstructed_generated_images, gender_reconstructed_generated , age_reconstructed_generated= cycle_pass(
    #             images=reconstructed_images,
    #             genders=gender_tensor_reconstruct,
    #             base_ages=age_tensor_reconstruct,
    #             change_age=False
    #         )

    #         loss_dict, loss = self.calc_loss2(
    #             input_images=images,
    #             generated_images=generated,
    #             generated_latents=latents,
    #             reconstruct_generated_images=reconstructed_generated_images,
    #             reconstruct_images=reconstructed_images,
    #             d_z_prior=d_z_prior,
    #             d_z=d_z,
    #             z=z,
    #             d_i_input=d_i_input,
    #             d_i_output=d_i_output,
    #             is_epoch_treshold=False,
    #             batch_number=batch_number,
    #         )
    #     return loss_dict, loss



    def forward_pass2(
        self,
        epoch_number,
        batch_number,
        images,
        ages,
        genders,
        epoch_treshold=50,
    ):

        def str_to_label(text: str):
            age, gender = text.split(".")
            age_tensor = torch.zeros(consts.NUM_AGES)
            gender_tensor = torch.zeros(consts.NUM_GENDERS)
            if epoch_number < epoch_treshold:
                age_tensor[int(age)] = 1
            else:
                age_tensor[np.random.randint(low=0, high=consts.NUM_AGES, size=1)] = 1
            gender_tensor[int(gender)] = 1
            return age_tensor, gender_tensor

        def cycle_pass(images, base_ages, genders , change_age):

            # if age_gender_labels == None:
            #     age_gender_labels = [
            #         (str_to_label(idx_to_class[l])) for l in list(labels.numpy())
            #     ]
            age_tensor_g = base_ages
            if change_age:
                age_tensor_g = torch.randint(10, 70, base_ages.shape)
            gender_tensor_g = genders
            # phase 1: getting style feature maps
            z = self.E(images)
            # only get NUM_AGES z's
            gender_tensor_expanded = gender_tensor.unsqueeze(1).repeat(
                1, consts.NUM_AGES + 1, 1
            )
            z_l = torch.cat([z, gender_tensor_expanded], dim=2)
            z = z[:, : consts.NUM_AGES + 1, :]
            # getting denoised image
            denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
                [z[:, 0, :]],
                return_latents=False,
                return_features=False,
                generated_image_length=256,
            )[0]
            # getting f_aw
            f_aw = self.Age_Modulator(
                denoised_images, age_tensor_g
            )  # f_aw = (batch , 512)
            f_aw = torch.unsqueeze(f_aw, dim=1)
            g_input = torch.cat([z_l, f_aw], dim=1)
            # feeding f_aw and w[0]to generator
            generated, latents = self.G(
                [g_input], return_latents=True, generated_image_length=128
            )
            return generated, gender_tensor_g , age_tensor_g

        # for 50 epoch we should train the network to map the images to the same age and get reconstruction loss
        # if epoch_number < epoch_treshold:
        images = images.to(self.device)
        base_age_tensor = ages.to(self.device)
        gender_tensor = genders.to(self.device)
        # if epoch_number < epoch_treshold:
        # target_age_tensor = torch.randint(10, 70, (consts.BATCH_SIZE, 1))
        age_tensor = base_age_tensor
        # phase 1: getting style feature maps
        z = self.E(images)
        # only get NUM_AGES z's
        gender_tensor_expanded = gender_tensor.unsqueeze(1).repeat(
            1, consts.NUM_AGES + 1, 1
        )
        z_l = torch.cat([z, gender_tensor_expanded], dim=2)
        z = z[:, : consts.NUM_AGES + 1, :]
        # getting denoised image
        denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
            [z[:, 0, :]],
            return_latents=False,
            return_features=False,
            generated_image_length=256,
        )[
            0
        ]
        # denoised_images = self.Freezed_stylegan_generator(  # denoised_images = (batch,3,1024,1024)
        #     self.mlp(z[:, 0, :]), return_latents=False, return_features=False
        # )
        # getting f_aw
        f_aw = self.Age_Modulator(
            denoised_images, age_tensor
        )  # f_aw = (batch , 512)
        f_aw = torch.unsqueeze(f_aw, dim=1)
        g_input = torch.cat([z_l, f_aw], dim=1)

        # feeding f_aw and w[0]to generator
        generated, latents = self.G(
            [g_input], return_latents=True, generated_image_length=128
        )
        z_prior = two_sided(
            torch.rand_like(z[:, 0, :], device=self.device)
        )  # [-1 : 1]

        # min max in uniform distribution
        d_z_prior = self.Dz(z_prior)
        d_z = self.Dz(z[:, 0, :])
        # minmax game in images
        d_i_input = self.Dimg(
            images, torch.cat([age_tensor, gender_tensor], dim=1), self.device
        )
        d_i_output = self.Dimg(
            generated, torch.cat([age_tensor, gender_tensor], dim=1), self.device
        )
        if epoch_number < epoch_treshold:
            # reconstructed image
            loss_dict, loss = self.calc_loss2(
                input_images=images,
                generated_images=generated,
                generated_latents=latents,
                reconstruct_generated_images=None,
                reconstruct_images=None,
                d_z_prior=d_z_prior,
                d_z=d_z,
                z=z,
                d_i_input=d_i_input,
                d_i_output=d_i_output,
                is_epoch_treshold=True,
                batch_number=batch_number,
            )
        else:
            # calculating for the cyclic losses
            reconstructed_images, gender_tensor_reconstruct , age_tensor_reconstruct = cycle_pass(images=generated, base_ages=base_age_tensor , genders=gender_tensor ,change_age=True)
            reconstructed_generated_images, gender_reconstructed_generated , age_reconstructed_generated= cycle_pass(
                images=reconstructed_images,
                genders=gender_tensor_reconstruct,
                base_ages=age_tensor_reconstruct,
                change_age=False
            )

            loss_dict, loss = self.calc_loss2(
                input_images=images,
                generated_images=generated,
                generated_latents=latents,
                reconstruct_generated_images=reconstructed_generated_images,
                reconstruct_images=reconstructed_images,
                d_z_prior=d_z_prior,
                d_z=d_z,
                z=z,
                d_i_input=d_i_input,
                d_i_output=d_i_output,
                is_epoch_treshold=False,
                batch_number=batch_number,
            )
        return loss_dict, loss



    def teach(
        self,
        utkface_path,
        batch_size=64,
        epochs=1,
        weight_decay=1e-5,
        lr=2e-4,
        should_plot=False,
        betas=(0.9, 0.999),
        valid_size=8000,
        where_to_save=None,
        models_saving="always",
    ):
        where_to_save = where_to_save or default_where_to_save()
        dataset = get_utkface_dataset(utkface_path)
        # getting the best dataset for the purpose:
        # dataset = Celeba_10000_dataset()
        valid_size = valid_size or batch_size
        valid_dataset, train_dataset = torch.utils.data.random_split(
            dataset, (valid_size, len(dataset) - valid_size)
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=15,
            drop_last=True,
        )
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

        input_output_loss = l1_loss
        nrow = round((2 * batch_size) ** 0.5)

        for optimizer in (self.eg_optimizer, self.dz_optimizer, self.di_optimizer):
            for param in ("weight_decay", "betas", "lr"):
                val = locals()[
                    param
                ]  # search the values by the name in this scope in function
                if val is not None:
                    optimizer.param_groups[0][
                        param
                    ] = val  # changes the optimizers value based params name
        loss_tracker = LossTracker(plot=should_plot)
        where_to_save_epoch = ""

        save_count = 0
        paths_for_gif = []

        for epoch in range(1, epochs + 1):
            where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch))
            try:
                if not os.path.exists(where_to_save_epoch):
                    os.makedirs(where_to_save_epoch)
                paths_for_gif.append(where_to_save_epoch)
                losses = defaultdict(lambda: [])
                self.train()  # move to train mode
                # for i, (images, labels) in enumerate(train_loader, 1):
                for i, (images, labels) in tqdm(
                    enumerate(train_loader, 1), total=len(train_loader)
                ):

                    # images = images.to(device=self.device)
                    # labels = torch.stack(
                    #     [
                    #         str_to_tensor(idx_to_class[l], normalize=True)
                    #         for l in list(labels.numpy())
                    #     ]
                    # )

                    # labels = labels.to(device=self.device)

                    # losses, loss = self.forward_pass(
                    #     idx_to_class=idx_to_class,
                    #     epoch_number=epoch,
                    #     batch_number=i,
                    #     images=images,
                    #     epoch_treshold=30,
                    # )
                    losses, loss = self.forward_pass(
                        idx_to_class=idx_to_class,
                        epoch_number=epoch,
                        batch_number=i,
                        labels=labels,
                        images=images,
                        epoch_treshold=30,
                    )
                    self.eg_optimizer.zero_grad()
                    self.dz_optimizer.zero_grad()
                    self.di_optimizer.zero_grad()
                    self.age_modulator_optimizer.zero_grad()
                    self.arc_face_loss_optimizer.zero_grad()  # for id loss (arcface loss) training
                    self.age_classifier_optimizer.zero_grad()

                    # Back prop on Encoder\Generator
                    # print(f"{loss=} , {losses = }  in batch: {i}")
                    loss.backward()

                    self.eg_optimizer.step()
                    self.dz_optimizer.step()
                    self.di_optimizer.step()
                    self.age_modulator_optimizer.step()
                    self.arc_face_loss_optimizer.step()  # for id loss (arcface loss) training
                    self.age_classifier_optimizer.step()

                    now = datetime.datetime.now()

                logging.info(
                    "[{h}:{m}[Epoch {e}] Loss: {t}".format(
                        h=now.hour, m=now.minute, e=epoch, t=losses.items()
                    )
                )
                print("----------------------------------------------")
                # print_timestamp(f"[Epoch {epoch:d}] Loss: {losses.item():f}")
                print(f"Loss: {losses.items()}")
                to_save_models = models_saving in ("always", "tail")
                cp_path = self.save(where_to_save_epoch, to_save_models=to_save_models)
                if models_saving == "tail":
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, "losses.png"))

                loss_tracker.append_many(
                    **{
                        k: torch.mean(torch.tensor(v, dtype=torch.float32))
                        for k, v in losses.items()
                    }
                )
                loss_tracker.plot()

                logging.info(
                    "[{h}:{m}[Epoch {e}] Loss: {l}".format(
                        h=now.hour, m=now.minute, e=epoch, l=repr(loss_tracker)
                    )
                )

            except KeyboardInterrupt:
                print_timestamp(
                    "{br}CTRL+C detected, saving model{br}".format(br=os.linesep)
                )
                if models_saving != "never":
                    cp_path = self.save(where_to_save_epoch, to_save_models=True)
                if models_saving == "tail":
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, "losses.png"))
                raise

        if models_saving == "last":
            cp_path = self.save(where_to_save_epoch, to_save_models=True)
        loss_tracker.plot()

    # def teach2(
    #     self,
    #     utkface_path,
    #     batch_size=64,
    #     epochs=1,
    #     weight_decay=1e-5,
    #     lr=2e-4,
    #     should_plot=False,
    #     betas=(0.9, 0.999),
    #     valid_size=8000,
    #     where_to_save=None,
    #     models_saving="always",
    # ):
    #     where_to_save = where_to_save or default_where_to_save()
    #     # dataset = get_utkface_dataset(utkface_path)
    #     # getting the best dataset for the purpose:
    #     dataset = Celeba_10000_dataset()
    #     valid_size = valid_size or batch_size
    #     valid_dataset, train_dataset = torch.utils.data.random_split(
    #         dataset, (valid_size, len(dataset) - valid_size)
    #     )

    #     train_loader = DataLoader(
    #         dataset=train_dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=15,
    #         drop_last=True,
    #     )
    #     idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    #     input_output_loss = l1_loss
    #     nrow = round((2 * batch_size) ** 0.5)

    #     for optimizer in (self.eg_optimizer, self.dz_optimizer, self.di_optimizer):
    #         for param in ("weight_decay", "betas", "lr"):
    #             val = locals()[
    #                 param
    #             ]  # search the values by the name in this scope in function
    #             if val is not None:
    #                 optimizer.param_groups[0][
    #                     param
    #                 ] = val  # changes the optimizers value based params name
    #     loss_tracker = LossTracker(plot=should_plot)
    #     where_to_save_epoch = ""

    #     save_count = 0
    #     paths_for_gif = []

    #     for epoch in range(1, epochs + 1):
    #         where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch))
    #         try:
    #             if not os.path.exists(where_to_save_epoch):
    #                 os.makedirs(where_to_save_epoch)
    #             paths_for_gif.append(where_to_save_epoch)
    #             losses = defaultdict(lambda: [])
    #             self.train()  # move to train mode
    #             # for i, (images, labels) in enumerate(train_loader, 1):
    #             for i, (images, ages, genders) in tqdm(
    #                 enumerate(train_loader, 1), total=len(train_loader)
    #             ):

    #                 # images = images.to(device=self.device)
    #                 # labels = torch.stack(
    #                 #     [
    #                 #         str_to_tensor(idx_to_class[l], normalize=True)
    #                 #         for l in list(labels.numpy())
    #                 #     ]
    #                 # )

    #                 # labels = labels.to(device=self.device)

    #                 losses, loss = self.forward_pass2(
    #                     idx_to_class=idx_to_class,
    #                     epoch_number=epoch,
    #                     batch_number=i,
    #                     images=images,
    #                     ages=ages,
    #                     genders=genders,
    #                     epoch_treshold=30,
    #                 )
    #                 self.eg_optimizer.zero_grad()
    #                 self.dz_optimizer.zero_grad()
    #                 self.di_optimizer.zero_grad()
    #                 self.age_modulator_optimizer.zero_grad()
    #                 self.arc_face_loss_optimizer.zero_grad()  # for id loss (arcface loss) training
    #                 self.age_classifier_optimizer.zero_grad()

    #                 # Back prop on Encoder\Generator
    #                 # print(f"{loss=} , {losses = }  in batch: {i}")
    #                 loss.backward()

    #                 self.eg_optimizer.step()
    #                 self.dz_optimizer.step()
    #                 self.di_optimizer.step()
    #                 self.age_modulator_optimizer.step()
    #                 self.arc_face_loss_optimizer.step()  # for id loss (arcface loss) training
    #                 self.age_classifier_optimizer.step()

    #                 now = datetime.datetime.now()

    #             logging.info(
    #                 "[{h}:{m}[Epoch {e}] Loss: {t}".format(
    #                     h=now.hour, m=now.minute, e=epoch, t=losses.items()
    #                 )
    #             )
    #             print("----------------------------------------------")
    #             # print_timestamp(f"[Epoch {epoch:d}] Loss: {losses.item():f}")
    #             print(f"Loss: {losses.items()}")
    #             to_save_models = models_saving in ("always", "tail")
    #             cp_path = self.save(where_to_save_epoch, to_save_models=to_save_models)
    #             if models_saving == "tail":
    #                 prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
    #                 remove_trained(prev_folder)
    #             loss_tracker.save(os.path.join(cp_path, "losses.png"))

    #             loss_tracker.append_many(
    #                 **{
    #                     k: torch.mean(torch.tensor(v, dtype=torch.float32))
    #                     for k, v in losses.items()
    #                 }
    #             )
    #             loss_tracker.plot()

    #             logging.info(
    #                 "[{h}:{m}[Epoch {e}] Loss: {l}".format(
    #                     h=now.hour, m=now.minute, e=epoch, l=repr(loss_tracker)
    #                 )
    #             )

    #         except KeyboardInterrupt:
    #             print_timestamp(
    #                 "{br}CTRL+C detected, saving model{br}".format(br=os.linesep)
    #             )
    #             if models_saving != "never":
    #                 cp_path = self.save(where_to_save_epoch, to_save_models=True)
    #             if models_saving == "tail":
    #                 prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
    #                 remove_trained(prev_folder)
    #             loss_tracker.save(os.path.join(cp_path, "losses.png"))
    #             raise

    #     if models_saving == "last":
    #         cp_path = self.save(where_to_save_epoch, to_save_models=True)
    #     loss_tracker.plot()


    def teach2(
        self,
        utkface_path,
        batch_size=64,
        epochs=1,
        weight_decay=1e-5,
        lr=2e-4,
        should_plot=False,
        betas=(0.9, 0.999),
        valid_size=8000,
        where_to_save=None,
        models_saving="always",
    ):
        where_to_save = where_to_save or default_where_to_save()
        # dataset = get_utkface_dataset(utkface_path)
        # getting the best dataset for the purpose:
        dataset = Celeba_10000_dataset()
        valid_size = valid_size or batch_size
        valid_dataset, train_dataset = torch.utils.data.random_split(
            dataset, (valid_size, len(dataset) - valid_size)
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=15,
            drop_last=True,
        )
        # idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

        input_output_loss = l1_loss
        nrow = round((2 * batch_size) ** 0.5)

        for optimizer in (self.eg_optimizer, self.dz_optimizer, self.di_optimizer):
            for param in ("weight_decay", "betas", "lr"):
                val = locals()[
                    param
                ]  # search the values by the name in this scope in function
                if val is not None:
                    optimizer.param_groups[0][
                        param
                    ] = val  # changes the optimizers value based params name
        loss_tracker = LossTracker(plot=should_plot)
        where_to_save_epoch = ""

        save_count = 0
        paths_for_gif = []

        for epoch in range(1, epochs + 1):
            where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch))
            try:
                if not os.path.exists(where_to_save_epoch):
                    os.makedirs(where_to_save_epoch)
                paths_for_gif.append(where_to_save_epoch)
                losses = defaultdict(lambda: [])
                self.train()  # move to train mode
                # for i, (images, labels) in enumerate(train_loader, 1):
                for i, (images, ages, genders) in tqdm(
                    enumerate(train_loader, 1), total=len(train_loader)
                ):

                    # images = images.to(device=self.device)
                    # labels = torch.stack(
                    #     [
                    #         str_to_tensor(idx_to_class[l], normalize=True)
                    #         for l in list(labels.numpy())
                    #     ]
                    # )

                    # labels = labels.to(device=self.device)

                    losses, loss = self.forward_pass2(
                        # idx_to_class=None,
                        epoch_number=epoch,
                        batch_number=i,
                        images=images,
                        ages=ages,
                        genders=genders,
                        epoch_treshold=30,
                    )
                    self.eg_optimizer.zero_grad()
                    self.dz_optimizer.zero_grad()
                    self.di_optimizer.zero_grad()
                    self.age_modulator_optimizer.zero_grad()
                    self.arc_face_loss_optimizer.zero_grad()  # for id loss (arcface loss) training
                    self.age_classifier_optimizer.zero_grad()

                    # Back prop on Encoder\Generator
                    # print(f"{loss=} , {losses = }  in batch: {i}")
                    loss.backward()

                    self.eg_optimizer.step()
                    self.dz_optimizer.step()
                    self.di_optimizer.step()
                    self.age_modulator_optimizer.step()
                    self.arc_face_loss_optimizer.step()  # for id loss (arcface loss) training
                    self.age_classifier_optimizer.step()

                    now = datetime.datetime.now()

                logging.info(
                    "[{h}:{m}[Epoch {e}] Loss: {t}".format(
                        h=now.hour, m=now.minute, e=epoch, t=losses.items()
                    )
                )
                print("----------------------------------------------")
                # print_timestamp(f"[Epoch {epoch:d}] Loss: {losses.item():f}")
                print(f"Loss: {losses.items()}")
                to_save_models = models_saving in ("always", "tail")
                cp_path = self.save(where_to_save_epoch, to_save_models=to_save_models)
                if models_saving == "tail":
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, "losses.png"))

                loss_tracker.append_many(
                    **{
                        k: torch.mean(torch.tensor(v, dtype=torch.float32))
                        for k, v in losses.items()
                    }
                )
                loss_tracker.plot()

                logging.info(
                    "[{h}:{m}[Epoch {e}] Loss: {l}".format(
                        h=now.hour, m=now.minute, e=epoch, l=repr(loss_tracker)
                    )
                )

            except KeyboardInterrupt:
                print_timestamp(
                    "{br}CTRL+C detected, saving model{br}".format(br=os.linesep)
                )
                if models_saving != "never":
                    cp_path = self.save(where_to_save_epoch, to_save_models=True)
                if models_saving == "tail":
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, "losses.png"))
                raise

        if models_saving == "last":
            cp_path = self.save(where_to_save_epoch, to_save_models=True)
        loss_tracker.plot()



    def _mass_fn(self, fn_name, *args, **kwargs):
        """Apply a function to all possible Net's components.

        :return:
        """
        for class_attr in dir(self):
            if not class_attr.startswith("_"):
                class_attr_value = getattr(self, class_attr)
                # print(f"Class attribute: {class_attr}, Type: {type(class_attr_obj)}")
                if hasattr(class_attr_value, fn_name) and callable(class_attr_value):
                    fn = getattr(class_attr_value, fn_name)
                    if callable(fn):
                        fn(*args, **kwargs)

    def to(self, device):
        self._mass_fn("to", device=device)

    def cpu(self):
        self._mass_fn("cpu")
        self.device = torch.device("cpu")

    def cuda(self):
        self._mass_fn("cuda")
        self.device = torch.device("cuda")

    def eval(self):
        """Move Net to evaluation mode.

        :return:
        """
        self._mass_fn("eval")

    def train(self):
        """Move Net to training mode.

        :return:
        """
        self._mass_fn("train")

    # def save(self, path, to_save_models=True):
    #     """Save all state dicts of Net's components.

    #     :return:
    #     """
    #     if not os.path.isdir(path):
    #         os.mkdir(path)

    #     saved = []
    #     if to_save_models:
    #         for class_attr_name in dir(self):
    #             if not class_attr_name.startswith("_"):
    #                 class_attr = getattr(self, class_attr_name)
    #                 if hasattr(class_attr, "state_dict"):
    #                     state_dict = class_attr.state_dict()
    #                     fname = os.path.join(
    #                         path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name)
    #                     )
    #                     torch.save(state_dict, fname)
    #                     saved.append(class_attr_name)

    #     if saved:  # if it's not None
    #         print_timestamp("Saved {} to {}".format(", ".join(saved), path))
    #     elif to_save_models:
    #         raise FileNotFoundError("Nothing was saved to {}".format(path))
    #     return path

    def save(self, path, to_save_models=True):
        """Save all state dicts of Net's components.

        :return:
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        saved = []
        # if to_save_models:
        #     for class_attr_name in dir(self):
        #         if not class_attr_name.startswith("_"):
        #             class_attr = getattr(self, class_attr_name)
        #             if hasattr(class_attr, "state_dict"):
        #                 state_dict = class_attr.state_dict()
        #                 fname = os.path.join(
        #                     path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name)
        #                 )
        #                 torch.save(state_dict, fname)
        #                 saved.append(class_attr_name)
        models = [self.E, self.Age_Modulator, self.G]
        models = {
            "AgeEncoder4Editing": self.E,
            "Age_Modulator": self.Age_Modulator,
            "Generator_Stylegan2": self.G,
        }
        for k, v in models.items():
            state_dict = v.state_dict()
            torch.save(state_dict, os.path.join(path, k) + ".pt")
            saved.append(v)
        if saved:  # if it's not None
            print_timestamp("Saved {} to {}".format(", ".join(saved), path))
        elif to_save_models:
            raise FileNotFoundError("Nothing was saved to {}".format(path))
        return path

    def load(self, path, slim=True):
        """Load all state dicts of Net's components.

        :return:
        """
        loaded = []
        for class_attr_name in dir(self):
            if (not class_attr_name.startswith("_")) and (
                (not slim) or (class_attr_name in ("E", "G"))
            ):
                class_attr = getattr(self, class_attr_name)
                fname = os.path.join(
                    path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name)
                )
                if hasattr(class_attr, "load_state_dict") and os.path.exists(fname):
                    class_attr.load_state_dict(torch.load(fname)())
                    loaded.append(class_attr_name)
        if loaded:
            print_timestamp("Loaded {} from {}".format(", ".join(loaded), path))
        else:
            raise FileNotFoundError("Nothing was loaded from {}".format(path))


def create_list_of_img_paths(pattern, start, step):
    result = []
    fname = pattern.format(start)
    while os.path.isfile(fname):
        result.append(fname)
        start += step
        fname = pattern.format(start)
    return result


def create_gif(img_paths, dst, start, step):
    BLACK = (255, 255, 255)
    WHITE = (255, 255, 255)
    MAX_LEN = 1024
    frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    corner = (2, 25)
    fontScale = 0.5
    fontColor = BLACK
    lineType = 2
    for path in img_paths:
        image = cv2.imread(path)
        height, width = image.shape[:2]
        current_max = max(height, width)
        if current_max > MAX_LEN:
            height = int(height / current_max * MAX_LEN)
            width = int(width / current_max * MAX_LEN)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(image, 50, 0, 0, 0, cv2.BORDER_CONSTANT, WHITE)
        cv2.putText(
            image, "Epoch: " + str(start), corner, font, fontScale, fontColor, lineType
        )
        image = image[..., ::-1]
        frames.append(image)
        start += step
    imageio.mimsave(dst, frames, "GIF", duration=0.5)
