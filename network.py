# utils
import consts
from datasets.datasets import Celeba_10000_dataset
from datasets.augmentations import AgeTransformer
# image  processing and logging
from utils.pipeline_utils import *
from tqdm import tqdm
import logging
import random
from collections import OrderedDict
# import imageio

import cv2
from PIL import Image
import numpy as np

# loss and optimizers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits_loss
from utils import common, train_utils
from criteria import id_loss
from configs import data_configs
from criteria.lpips.lpips import LPIPS
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


# modules
# from models.race_net import Race_net
# from models.stylegan2.model import Generator
# from models.encoders.psp_encoders import AgeEncoder4Editing
# from models.feature_extractor.feature_extractor import Feature_extractor
# from models.age_modulator import resnet
# from torchvision.models import resnet18 , resnet34
from models.psp import pSp
# path configs
from configs.paths_config import model_paths
from criteria.aging_loss import AgingLoss
from criteria.w_norm import WNormLoss

# debugging
torch.autograd.set_detect_anomaly(True)
from torchsummary import summary
import os
import math

# ========================= RageGAN ================================

class Net(object):
    def __init__(self , opts):
        # resources
        # self.device = "cpu"
        self.device = "cuda"
        self.opts = opts
        # # utils modules
        self.age_transformer = AgeTransformer(target_age=self.opts.target_age)
        # # Modules
        self.net = pSp(self.opts).to(self.device)
        for p in self.net.encoder.parameters():
          p.requires_grad= True  # Should be True
        # # optimizers
        self.optimizer_age = Adam(params = self.net.encoder.parameters(),
                                lr =  0.0001,
                                # betas = (self.opts.b1 , self.opts.b2),
                                # weight_decay = self.opts.weight_decay
                                )
        param =[]
        for race_mixer in self.net.race_mixers:
            param.extend(race_mixer.parameters())
        for style in self.net.pretrained_encoder.styles:
            param.extend(style.parameters())
        param.extend(self.net.feature_mixer.parameters())
        self.optimizer_reconstruct = Adam(params =param,
                                lr =  self.opts.learning_rate,
                                betas = (self.opts.b1 , self.opts.b2),
                                weight_decay = self.opts.weight_decay
                              )

        # Initialize loss
        if opts.mode == "train":
            self.mse_loss = nn.MSELoss().to(self.device).eval()
            if self.opts.lpips_lambda > 0:
                self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
            if self.opts.id_lambda > 0:
                self.id_loss = id_loss.IDLoss().to(self.device).eval()
            if self.opts.w_norm_lambda > 0:
                self.w_norm_loss = WNormLoss(opts=self.opts)
            if self.opts.aging_lambda > 0:
                self.aging_loss = AgingLoss(self.opts)
            if self.opts.race_lambda > 0:
                self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device).eval()
        # # self.cpu()  # initial, can later move to cuda
        # self.cuda()  # initial, can later move to cuda
        
    def __get_keys(d, name):
        # if 'state_dict' in d:
        d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt

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
        train_size = 2,
        where_to_save=None,
        models_saving="always",
    ):
        where_to_save = where_to_save
        # dataset = get_utkface_dataset(utkface_path)
        # getting the best dataset for the purpose:
        dataset = Celeba_10000_dataset()
        
        valid_size = valid_size or batch_size
        # valid_dataset, train_dataset = torch.utils.data.random_split(
        #     dataset, (len(dataset) - self.opts.train_size, self.opts.train_size)
        # )
        valid_dataset, train_dataset = torch.utils.data.random_split(
            dataset, (0 , len(dataset))
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=15,
            drop_last=True,
        )

        input_output_loss = l1_loss
        nrow = round((2 * batch_size) ** 0.5)

        loss_tracker = LossTracker(plot=should_plot)
        where_to_save_epoch = ""

        save_count = 0
        paths_for_gif = []

        self.net.train()  # move to train mode
        for epoch in range(1, epochs + 1):
            where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch))
            try:
                if not os.path.exists(where_to_save_epoch):
                    os.makedirs(where_to_save_epoch)
                paths_for_gif.append(where_to_save_epoch)
                # for i, (images, labels) in enumerate(train_loader, 1):
                for i, (images, ages, genders) in tqdm(
                    enumerate(train_loader, 1), total=len(train_loader)
                ):
                    torch.cuda.empty_cache()
                    ages= ages.unsqueeze(dim = 1).to(self.device)
                    genders= genders.unsqueeze(dim = 1).to(self.device)

                    # self.optimizer_reconstruct.zero_grad()
                    self.optimizer_age.zero_grad()
                    generated_images ,generated_latents, target_ages , no_aging = self.forward(
                        epoch_number=epoch,
                        batch_number=i,
                        images=images,
                        ages=ages,
                        genders=genders,
                        no_aging="permitted"
                    )
                    target_ages.to(self.device)
                    loss , loss_dict , id_logs = self.calc_loss(images, images, generated_images, generated_latents,
														  target_ages=target_ages,
														  input_ages=ages,
														  no_aging=no_aging,
														  data_type="real")
                    # print(f"{loss.shape=}")
                    # loss.backward(retain_graph = True)
                    loss.backward()
                    # if self.is_source_image:
                    #     self.optimizer_reconstruct.step()
                    # else:
                    #     self.optimizer_age.step()
                    # self.optimizer_age.step()
                    generated_images_clone = generated_images.clone().detach().requires_grad_(True)
                    input_ages_clone = ages.clone().detach().float().requires_grad_(True)
                    input_genders_clone = genders.clone().detach().float().requires_grad_(True)
                    reconstructed_input_images ,reconstructed_generated_latents, reconstructed_target_ages , reconstructed_no_aging = self.forward(
                        epoch_number=epoch,
                        batch_number=i,
                        images=generated_images_clone,
                        ages=input_ages_clone,
                        genders=input_genders_clone,
                        no_aging="not permitted"
                    )
                    loss2 , loss_dict , id_logs = self.calc_loss(images, images, reconstructed_input_images, reconstructed_generated_latents,
                                        target_ages=reconstructed_target_ages,
                                        input_ages=input_ages_clone,
                                        no_aging=reconstructed_no_aging,
                                        data_type="real")
                    
                    loss = loss2
                    loss.backward()              
                    # self.optimizer_reconstruct.step()
                    self.optimizer_age.step()
                    now = datetime.datetime.now()

                logging.info(
                    "[{h}:{m}[Epoch {e}] Loss: {t}".format(
                        h=now.hour, m=now.minute, e=epoch, t=loss_dict.items()
                    )
                )
                print("----------------------------------------------")
                # print_timestamp(f"[Epoch {epoch:d}] Loss: {losses.item():f}")
                print(f"Loss: {loss_dict.items()}")
                to_save_models = models_saving in ("always", "tail")
                cp_path = self.save(where_to_save_epoch, to_save_models=to_save_models)
                if models_saving == "tail":
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, "losses.png"))

                loss_tracker.append_many(
                    **{
                        k: torch.mean(torch.tensor(v, dtype=torch.float32))
                        for k, v in loss_dict.items()
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

    def forward(
        self,
        epoch_number,
        batch_number,
        images,
        ages,
        genders,
        no_aging = "permitted"
    ):

        # print(f"{images.shape=} , {ages.shape=} , {genders.shape=}")
        def cycle(images , ages , genders, no_aging = no_aging):
            images = images.to(self.device)
            ages = ages.to(self.device)
            genders = genders.to(self.device)
            # perform no aging in 33% of the time
            no_aging = random.random() <= (1. / 3) if no_aging == "permitted" else True
            input_ages= ages/100
            if no_aging:
                x_input = self.__set_target_to_source(x=images, input_ages=input_ages)
                self.is_source_image = True
            else:
                x_input = [self.age_transformer(img.cpu()).to(self.device) for img in images]
                self.is_source_image = False
            x_input= torch.stack(x_input, dim= 0)
            target_ages = x_input[:, -1, 0, 0].unsqueeze(dim=1)
            # from  torchvision.transforms import transforms
            # tr = transforms.ToPILImage()
            # in_image = tr(images[0])
            # in_image.save("/content/in.jpg")
            generated_images , generated_latents = self.net.forward(x_input ,return_latents=True , resize=True)
            # pil_image = tr(generated_images[0])
            # pil_image.save("/content/out.jpg")
            return generated_images , generated_latents , target_ages , no_aging
        
        # perform cyclic pass
        return cycle(images, ages, genders, no_aging=no_aging)

    def __set_target_to_source(self, x, input_ages):
        # print(f"{input_ages}")
        return [torch.cat((img, age* torch.ones((1, img.shape[1], img.shape[2])).to(self.device)))
                for img, age in zip(x, input_ages)]

    def calc_loss(self, x, y, y_hat, latent, target_ages, input_ages, no_aging, data_type="real"):
        
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = y_hat.to(self.device)
        input_ages = input_ages/100
        loss_dict = defaultdict(lambda: [])
        id_logs = []
        loss = 0.0
        
        if self.opts.id_lambda > 0:
            weights = None
            if self.opts.use_weighted_id_loss:  # compute weighted id loss only on forward pass
                target_ages.to(self.device)
                input_ages.to(self.device)
                # print(f"{target_ages.shape=} , {input_ages.shape=}")
                age_diffs = torch.abs(target_ages - input_ages)
                # if self.is_source_image:
                #     weights = torch.full_like(age_diffs , 1.0)
                # else:
                #     # weights = age_diffs/100
                weights = train_utils.compute_cosine_weights(x=age_diffs/100)
                # weights = np.abs(weights.float().cpu().detach().numpy())
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, label=data_type, weights=weights)
            # print(f"{loss_id.shape=}")
            loss_dict[f'loss_id'].append(loss_id)
            loss_dict[f'id_improve'].append(sim_improvement)
            loss = loss_id * self.opts.id_lambda
            # print(loss.shape)
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict[f'loss_l2'].append(float(loss_l2))
            if data_type == "real" and not no_aging:
                l2_lambda = self.opts.l2_lambda_aging
            else:
                l2_lambda = self.opts.l2_lambda
            loss += loss_l2 * l2_lambda
            # print(loss.shape)
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict[f'loss_lpips'].append(float(loss_lpips))
            if data_type == "real" and not no_aging:
                lpips_lambda = self.opts.lpips_lambda_aging
            else:
                lpips_lambda = self.opts.lpips_lambda
            loss += loss_lpips * lpips_lambda
        # print(loss.shape)
        # if self.opts.lpips_lambda_crop > 0:
        #     loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
        #     loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
        #     loss += loss_lpips_crop * self.opts.lpips_lambda_crop
        # if self.opts.l2_lambda_crop > 0:
        #     loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
        #     loss_dict['loss_l2_crop'] = float(loss_l2_crop)
        #     loss += loss_l2_crop * self.opts.l2_lambda_crop
        if self.opts.w_norm_lambda > 0:
            self.latent_avg = self.net.latent_avg
            # print(f"{self.latent_avg=}")
            loss_w_norm = self.w_norm_loss(latent, latent_avg=self.latent_avg)
            loss_dict[f'loss_w_norm'].append(float(loss_w_norm))
            loss += loss_w_norm * self.opts.w_norm_lambda
            # print(loss.shape)
        if self.opts.aging_lambda > 0:
            aging_loss, id_logs = self.aging_loss(y_hat, y, target_ages, id_logs, label=data_type)
            loss_dict[f'loss_aging'].append(float(aging_loss))
            loss += aging_loss * self.opts.aging_lambda
            # print(loss.shape)
        if self.opts.race_lambda >0:
            # hold = torch.zeros_like(self.net.race_net(x)).to(self.device)
            # hold[:,torch.argmax(self.net.race_net(x), dim = -1)]
            # race_loss = self.cross_entropy_loss(self.net.race_net(y_hat), hold)
            race_loss = F.mse_loss(self.net.race_net.get_features(self.convert_for_race_loss(y_hat)), self.net.race_net.get_features(self.convert_for_race_loss(y)))
            loss_dict["race_loss"].append(race_loss)
            loss += race_loss * self.opts.race_lambda
        if data_type == "cycle":
            loss = loss * self.opts.cycle_lambda
            # print(loss.shape)
        loss_dict[f'loss'].append(loss)
        return loss, loss_dict, id_logs

    def convert_for_race_loss(self,x):
        def unnormalize(tensor, mean, std):
            mean = torch.tensor(mean).to(self.device).view(1, 3, 1, 1)
            std = torch.tensor(std).to(self.device).view(1, 3, 1, 1)
            return tensor * std + mean

        x = unnormalize(x , [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        x = F.interpolate(x[:,:,60:200,70:230],size=(224, 224), mode='bilinear', align_corners=False)
        trans = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        y = []
        for x_ in x:
            y.append(trans(x_))
        y  = torch.stack(y, dim=0)
        return y.to(self.device)

    def test(self,
        images,
        ages,
        genders,
        no_aging = "permitted"):

        dev = "cuda"
        self.E.to(dev)
        self.age_modulator.to(dev)
        self.race_modulator.to(dev)
        self.race_net.to(dev)
        self.feature_extractor.to(dev)
        images = images.to(dev)
        ages = ages.to(dev)
        genders = genders.to(dev)
        # perform no aging in 33% of the time
        no_aging = random.random() <= (1. / 3) if no_aging == "permitted" else True
        # no_aging = True
        if no_aging:
            x_input = self.__set_target_to_source(x=images, input_ages=ages)
        else:
            x_input = [self.age_transformer(img.cpu()).to(self.device) for img in images]
        x_input= torch.stack(x_input, dim= 0)
        target_ages = x_input[:, -1, 0, 0].unsqueeze(dim=1)
        # print(f"{x_input.shape=}")
        latent_codes = self.E(x_input)   #(batch , 18 , 512)
        # print(f"{latent_codes.shape=}")
        # making the feature_extractor input
        image_features = self.feature_extractor(images)
        # print(f"{image_features.shape=}")
        # making the age_modulator inputs (inputing gender and age in a same modulator)
        age_tensor = [torch.zeros((1, 102)).type(torch.long) for age in ages]
        for i in range(ages.shape[0]):
            age_tensor[i][0 , int(ages[i][0].cpu())] += 1
            if genders[i, 0] == 0:
                age_tensor[i][0,-1] += 1
            else:
                age_tensor[i][0,-2] += 1
        age_tensor = torch.cat(age_tensor, dim =0).to(self.device)
        # reshaping the latent_codes
        latnet_codes_with_features = latent_codes + image_features.unsqueeze(dim=1).repeat(1,18,1)
        latent_codes_expanded = torch.cat([latent_codes , latnet_codes_with_features], dim =-1).reshape(latent_codes.shape[0], latent_codes.shape[1], 32, 32).to(self.device)
        # print(f"{latent_codes_expanded.shape=}")
        age_modulator_output = self.age_modulator(latent_codes_expanded, age_tensor.type(torch.float32))
        # print(f"{age_modulator_output.shape=}")
        # making the race_modulators input and the race features
        race_modulator_input = self.race_net(images)
        race_modulator_output  = self.race_modulator(latent_codes_expanded, race_modulator_input)
        # print(f"{race_modulator_output.shape=}")
        race_features =  self.race_net.get_features(images)
        # print(f"{race_features.shape=}")
        # combining the outputs of modulators
        lambda_race_features = 1
        lambda_race_modulator_output = 1
        lambda_age_modulator_output = 1
        lambda_image_features = 1
        final_features = (lambda_race_features * race_features +
                        lambda_race_modulator_output * race_modulator_output +
                        lambda_age_modulator_output * age_modulator_output +
                        lambda_image_features * image_features)
        # print(f"{final_features.shape=}")
        final_features.cpu()
        generator_input = final_features.unsqueeze(dim = 1).repeat(1,18,1) + latent_codes
        self.E.cpu()
        self.age_modulator.cpu()
        self.race_modulator.cpu()
        self.race_net.cpu()
        self.feature_extractor.cpu()
        print(torch.cuda.memory_cached())
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(torch.cuda.memory_cached())
        print(f"{generator_input.shape=}")
        return  self.G([generator_input.to("cuda")],return_latents= True)

    def save(self, path, to_save_models=True):
        """Save all state dicts of Net's components.

        :return:
        """
        
        if not os.path.isdir(path):
            os.mkdir(path)

        saved = []
        # models = {
        #     # "feature_mixer": self.net.feature_mixer,
        #     "encoder": self.net.encoder,
        # }
        models = {
            "feature_mixer": self.net.feature_mixer,
            "pretrained_encoder": self.net.pretrained_encoder,
            "race_mixer1": self.net.race_mixers[0],
            "race_mixer2": self.net.race_mixers[1],
            "race_mixer3": self.net.race_mixers[2]
        }
        for k, v in models.items():
            state_dict = v.state_dict()
            torch.save(state_dict, os.path.join(path, k) + ".pt")
            saved.append(k)
        if saved:  # if it's not None
            print_timestamp("Saved {} to {}".format(", ".join(saved), path))
        elif to_save_models:
            raise FileNotFoundError("Nothing was saved to {}".format(path))
        return path