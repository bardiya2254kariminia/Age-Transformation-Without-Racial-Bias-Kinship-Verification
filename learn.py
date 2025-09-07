import torch
from torchvision.transforms import transforms
from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


sys.path.append(".")
sys.path.append("..")

from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp

img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
tr = transforms.ToPILImage()
model_path = "/content/drive/MyDrive/sam_ffhq_aging.pt"
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
pprint.pprint(opts)
# update the training options
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')
image_path = "/content/download (3).jpg"
original_image = Image.open(image_path).convert("RGB")
original_image.resize((256, 256))
input_image = img_transforms(original_image)
# we'll run the image on multiple target ages 
target_ages = [30]
age_transformer = [AgeTransformer(target_age=age) for age in target_ages][0]
with torch.no_grad():
    input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
    input_image_age = torch.stack(input_image_age)
    result_batch = net(input_image_age.to("cuda").float(), randomize_noise=False, resize=False)
    print(f"{result_batch.shape=}")
    result_image = tensor2im(result_batch.squeeze(dim=0))
result_image.save("/content/out.jpg")
    



