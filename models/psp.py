"""
This file defines the core research contribution
"""
import copy
from argparse import Namespace

import torch
from torch import nn
import math

from configs.paths_config import model_paths
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from models.feature_extractor.feature_extractor import Feature_extractor
from models.race_net import Race_net
from models.age_modulator import resnet
from torchvision.models import resnet18 , resnet34
import consts
import torch.nn.functional  as F
from torchvision.transforms import transforms

class feature_mixer(nn.Module):
    """
        a feature mixing modules which gave (batch , in_c , h , w)
        ans output is (batch ,1,  h , w)
    
    """

    def __init__(self, opts, in_channel=2):
        super(feature_mixer,self).__init__()
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        self.opts = opts
		
        # adding modules to encoder
        encoder_channels = [in_channel ,32 ,16 , 8]
        for i, (in_c, out_c) in enumerate(zip(encoder_channels[:-1] , encoder_channels[1:]),start=1):
            self.encoder_list.add_module(
                f"conv_{i}", nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c , 
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
            self.encoder_list.add_module(
                f"bn_{i}" , nn.BatchNorm2d(num_features=out_c)
            )
            self.encoder_list.add_module(
                f"relu_{i}" , nn.LeakyReLU(negative_slope=0.2, inplace=False)
            )
            self.encoder_list.add_module(
                f"maxpool_{i}" , nn.MaxPool2d(kernel_size=3,
                                              stride=2,
                                              padding=1)
            )
        
        self.encoder_list.add_module(
            "adaptive_pooling" , nn.AdaptiveAvgPool2d((1, 128))
        )
        self.encoder = nn.Sequential(*self.encoder_list)
        # adding the moduel to decoder
        decoder_channels  = [8, 16, 32, 1]
        for i ,(in_c, out_c) in enumerate(zip(decoder_channels[:-1] , decoder_channels[1:]), start=1):
            self.decoder_list.add_module(
                f"deconv_{i}", nn.ConvTranspose2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
            self.decoder_list.add_module(
                f"relu_{i}", nn.LeakyReLU(
                    negative_slope=0.2
                )
            )
            if i != 4:
                self.decoder_list.add_module(
                    f"upsample_{i}", nn.Upsample(
                        mode="bilinear",
                        scale_factor=2,
                        align_corners=True
                    )
                )
        
        self.decoder_list.add_module(
            "upsample_4" , nn.Upsample(
                        mode="bilinear",
                        size=(18,512),
                        align_corners=True
                    )
        )
        self.decoder = nn.Sequential(*self.decoder_list)
        self.scaler= nn.Parameter(0.4 * torch.ones(18,512).cuda())
		
    
    def forward(self,x):
        codes = x[:,0,:,:]
        psp_code = x[:,1,:,:]
        out = self.encoder(x)
        out = self.decoder(out)
        return self.scaler * (codes + psp_code) + out.squeeze(dim=1)


class race_mixer(nn.Module):
    def __init__(self , in_c ,spatial , opts):
        super(race_mixer,self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_c,
                                           out_channels=in_c,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        self.opts = opts
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        channels = [2 , spatial , int(spatial/2)]
        for i , (in_chan ,out_chan) in enumerate(zip(channels[:-1] ,channels[1:]),start=1):
            self.encoder_list.add_module(
                f"conv_{i}" , nn.Conv2d(
                    in_channels=in_chan,
                    out_channels=out_chan,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            self.encoder_list.add_module(
                f"leaky_relu_{i}" , nn.LeakyReLU(negative_slope=0.2)
            )
        de_channels =[int(spatial/2),spatial ,1]
        for i , (in_chan ,out_chan) in enumerate(zip(de_channels[:-1] ,de_channels[1:]),start=1):
            self.decoder_list.add_module(
                f"deconv_{i}" , nn.ConvTranspose2d(
                    in_channels=in_chan,
                    out_channels=out_chan,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            self.decoder_list.add_module(
                f"leaky_relu_{i}" , nn.LeakyReLU(negative_slope=0.2)
            )
        self.encoder = nn.Sequential(*self.encoder_list)
        self.decoder = nn.Sequential(*self.decoder_list)
        self.scaler = nn.Parameter(0.6* torch.ones(1))
	
		
    def forward(self,x ,y):
        b ,c,h,w = y.shape
        x = self.upsample(x)
        x = x.reshape(b , c, h*w)
        y = y.reshape(b , c, h*w)
        input = torch.cat([x.unsqueeze(dim=1), y.unsqueeze(dim=1)] ,dim=1)
        latent = self.encoder(input)
        output= self.decoder(latent)
        output = output.squeeze(dim=1).reshape((b,c,h,w))
        # print(f"{y.shape=}")
        return output + self.scaler*y


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		self.device = "cuda"
		self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		self.decoder = Generator(self.opts.output_size, 512, 8)
		self.feature_mixer = feature_mixer(opts = opts)
		# self.set_feature_extractor()
		# self.set_race_modulator(pretrained=True, freeze=False)
		self.race_mixers = [race_mixer(in_c=128 , spatial=16 , opts = opts).to("cuda"),
							race_mixer(in_c=256 , spatial=32 , opts = opts).to("cuda"),  
							race_mixer(in_c=512 , spatial=64 , opts = opts).to("cuda")]
		self.race_net = Race_net(device=self.device).to(self.device)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def set_encoder(self):
		return psp_encoders.GradualStyleEncoder(50, 'ir_se', self.n_styles, self.opts)


	def set_age_modulator(self, pretrained=True, freeze=False):
		self.age_modulator = resnet.create_resnet(
			n="18_modified",
			# lstm_size=consts.NUM_AGES,
			lstm_size=102,
			emb_size=32,
			use_pretrained = False
		)
		if consts.RESNET_TYPE == 18:
			pretrained_res = resnet18(pretrained=pretrained).to(self.device)
		else:
			pretrained_res = resnet34(pretrained=True).to(self.device)

		model_state_dict = self.age_modulator.state_dict()
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
					for narg in ("bn", "running", "num_batches", "downsample", "fc", "conv1"):
						if narg in arg:
							flag = True
							break
					if flag:
						break
				if not flag:
					model_state_dict[key] = pretrained_model_state_dict[key]
			self.age_modulator.load_state_dict(model_state_dict)

			for name, module in self.age_modulator.named_modules():
				if "bn" not in name:
					for param in module.parameters():
						param.requires_grad = False

		if freeze:
			self.age_modulator.eval()
			for param in self.age_modulator.parameters():
				param.requires_grad = False

		self.age_modulator.to(self.device)

	def set_race_modulator(self, pretrained=True, freeze=False):
		self.race_modulator = resnet.create_resnet(
			n=18,
			# lstm_size=consts.NUM_AGES,
			lstm_size=9,
			emb_size=256,
			use_pretrained = False
		)
		if consts.RESNET_TYPE == 18:
			pretrained_res = resnet18(pretrained=pretrained).to(self.device)
		else:
			pretrained_res = resnet34(pretrained=pretrained).to(self.device)

		model_state_dict = self.race_modulator.state_dict()
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
					for narg in ("bn", "running", "num_batches", "downsample", "fc", "conv1"):
						if narg in arg:
							flag = True
							break
					if flag:
						break
				if not flag:
					model_state_dict[key] = pretrained_model_state_dict[key]
			self.race_modulator.load_state_dict(model_state_dict)

			for name, module in self.race_modulator.named_modules():
				if "bn" not in name:
					for param in module.parameters():
						param.requires_grad = False

		if freeze:
			self.race_modulator.eval()
			for param in self.race_modulator.parameters():
				param.requires_grad = False
		self.race_modulator.to(self.device)
		self.race_modulator.load_state_dict(torch.load("/content/artifacts/epoch20/race_modulator.pt", map_location="cpu"))
    

	def set_feature_extractor(self, pretrained = True , freeze = True):
		print(pretrained)
		self.feature_extractor  = Feature_extractor(pretrained=pretrained).to(self.device)
		if freeze:
			self.feature_extractor.eval()
			for param in self.feature_extractor.parameters():
				param.requires_grad = False
		self.feature_extractor.load_state_dict(torch.load("/content/artifacts/epoch20/feature_extractor.pt",map_location="cpu"))



	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print(f'Loading SAM from checkpoint: {self.opts.checkpoint_path}')
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			# Load the adjusted weights into the model
			self.encoder.load_state_dict(self.__get_keys(ckpt, 'encoder'), strict=False)
			self.decoder.load_state_dict(self.__get_keys(ckpt, 'decoder'), strict=True)
			# self.feature_mixer.load_state_dict(torch.load("/content/artifacts/epoch10/feature_mixer.pt", map_location=torch.device("cpu")), strict=True)
			if self.opts.start_from_encoded_w_plus:
				self.pretrained_encoder = self.__get_pretrained_psp_encoder()
				self.pretrained_encoder.load_state_dict(self.__get_keys(ckpt, 'pretrained_encoder'), strict=True)
			self.__load_latent_avg(ckpt)
			# for  param in  self.encoder.parameters():
			# 	param.requires_grad = False
			# for  param in  self.decoder.parameters():
			# 	param.requires_grad = False
			# for  param in  self.pretrained_encoder.parameters():
			# 	param.requires_grad = False
			# for  param in  self.race_net.parameters():
			# 	param.requires_grad = False
			# for  param in  self.feature_mixer.parameters():
			# 	param.requires_grad = False
			# self.decoder.eval()
			# self.pretrained_encoder.eval()
			
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			# Transfer the RGB input of the irse50 network to the first 3 input channels of SAM's encoder
			if self.opts.input_nc != 3:
				shape = encoder_ckpt['input_layer.0.weight'].shape
				altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
				altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
				encoder_ckpt['input_layer.0.weight'] = altered_input_layer
			self.encoder.load_state_dict(encoder_ckpt, strict=False)
			print(f'Loading decoder weights from pretrained path: {self.opts.stylegan_weights}')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
			self.__load_latent_avg(ckpt, repeat=self.n_styles)
			if self.opts.start_from_encoded_w_plus:
				self.pretrained_encoder = self.__load_pretrained_psp_encoder()
				self.pretrained_encoder.eval()

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
				inject_latent=None, return_latents=False, alpha=None, input_is_full=False):
		if input_code:
			codes = x
		else:
      
			# codes = self.encoder(x)
			codes = self.get_rage_out_age_encoder(x)
			# codes = codes + self.latent_avg
      
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				codes = codes + self.latent_avg
			# normalize with respect to the latent of the encoded image of pretrained pSp encoder
			elif self.opts.start_from_encoded_w_plus:
				with torch.no_grad():
					encoded_latents = self.pretrained_encoder(x[:, :-1, :, :])
					# encoded_latents = self.get_rage_out(x[:, :-1, :, :])
					encoded_latents = encoded_latents + self.latent_avg
				# codes = codes + encoded_latents

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0


		features = torch.stack([codes , encoded_latents], dim=1).to(self.device)
		decoder_input= self.feature_mixer(features).to(self.device)

		input_is_latent = (not input_code) or (input_is_full)
		# images, result_latent = self.decoder([codes],
		images, result_latent = self.decoder([decoder_input],
											 input_is_latent=input_is_latent,
											 randomize_noise=randomize_noise,
											 return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None

	def __get_pretrained_psp_encoder(self):
		opts_encoder = vars(copy.deepcopy(self.opts))
		opts_encoder['input_nc'] = 3
		opts_encoder = Namespace(**opts_encoder)
		encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.n_styles, opts_encoder)
		return encoder

	def __load_pretrained_psp_encoder(self):
		print(f'Loading pSp encoder from checkpoint: {self.opts.pretrained_psp_path}')
		ckpt = torch.load(self.opts.pretrained_psp_path, map_location='cpu')
		encoder_ckpt = self.__get_keys(ckpt, name='encoder')
		encoder = self.__get_pretrained_psp_encoder()
		encoder.load_state_dict(encoder_ckpt, strict=False)
		return encoder

	def convert_for_race_mixers(self, x):
		def unnormalize(tensor, mean, std):
			mean = torch.tensor(mean).to("cuda").view(1, 3, 1, 1)
			std = torch.tensor(std).to("cuda").view(1, 3, 1, 1)
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
		return torch.stack(y, dim=0).to("cuda")

	def get_rage_out(self,x):
		rc1 , rc2, rc3 =  self.race_net.get_c(self.convert_for_race_mixers(x))
		rc1 = F.interpolate(rc1,size=(32,32), mode='bilinear', align_corners=False)
		rc2 = F.interpolate(rc2,size=(32,32), mode='bilinear', align_corners=False)
		rc3 = F.interpolate(rc3,size=(32,32), mode='bilinear', align_corners=False)
		return self.pretrained_encoder(x ,rc1,rc2,rc3 ,  self.race_mixers)
	
	def get_rage_out_age_encoder(self,x):
		rc1 , rc2, rc3 =  self.race_net.get_c(self.convert_for_race_mixers(x[:,:-1,:,:]))
		rc1 = F.interpolate(rc1,size=(32,32), mode='bilinear', align_corners=False)
		rc2 = F.interpolate(rc2,size=(32,32), mode='bilinear', align_corners=False)
		rc3 = F.interpolate(rc3,size=(32,32), mode='bilinear', align_corners=False)
		return self.encoder(x ,rc1,rc2,rc3 ,  self.race_mixers)
	
	@staticmethod
	def __get_keys(d, name):
		if 'state_dict' in d:
			d = d['state_dict']
		d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
		return d_filt
