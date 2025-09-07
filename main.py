import network
import json
import consts
import logging
import os

os.system("nvidia-smi")
import re
import numpy as np
import argparse
import sys
import random
import datetime
import torch
from utils.pipeline_utils import *
from torchvision.datasets.folder import pil_loader
import gc
import torch
from cmd_options import Keyboared
import models
from torchsummary import summary

gc.collect()
import nextcloud_client
import gdown
import zipfile
import os
# import shutil




# fro test_single
import json
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms
from models.psp import pSp
from utils.common import tensor2im
import argparse
from argparse import Namespace
import  os
import pandas as pd
from  configs.paths_config import dataset_paths


def test_single(net, out_dir , annotation_name,image_folder_name):
    img_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # with open("SAM/config.json", "r") as fp:
    #     opts = json.load(fp)
    # print(opts)
    # opts = argparse.Namespace(**opts)

    # net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    ages = [20,30,40,50,60,70,80]
    # ages = [20]

    # os.makedirs("Ragegan_out", exist_ok=True)
    # out_dir = "Ragegan_out_with_Race"
    annotations = pd.read_csv(dataset_paths[annotation_name])
    indexes = range(len(annotations))
    for age in ages:
        print(f"on {age=} :")
        out_subdir =os.path.join(out_dir, str(age)) 
        os.makedirs(out_subdir , exist_ok=True)
        # for index in range(1):
        for index in range(len(annotations)):
            # try:
            image = Image.open(os.path.join(dataset_paths[image_folder_name],annotations.iloc[index , 1]))
            image_tensor = img_transforms(image)
            image_with_age = torch.cat((image_tensor , (age/100)*torch.ones(1,256,256))).unsqueeze(dim=0)
            out_image_tensor = net(image_with_age.to("cuda") ,resize=True)
            # print(type(out_image_tensor))
            # print(f"{out_image_tensor.shape=}")
            out_image = tensor2im(out_image_tensor.squeeze(dim=0).detach().cpu())
            out_image.save(os.path.join(out_subdir , annotations.iloc[index , 1]))
            # except:
            #     continue
        print(len(os.listdir(out_subdir)))

def getting_google_drive_data():
    config = {
        "pretrained_model": rf"https://drive.google.com/file/d/1-dmN_Pe1_5z8G5aD4zEUDOExDBlt8q1z/view",
        "CACD_UTKFACE_Dataset": rf"https://drive.google.com/file/d/1--mrHW8QrbGSB92YDoWz5KgFXCfeF2GP/view",
    }

    # gettting the uids and setting the output paths
    pretrained_models_uid = rf"1-dmN_Pe1_5z8G5aD4zEUDOExDBlt8q1z"
    pretrained_models_url = rf"https://drive.google.com/uc?id={pretrained_models_uid}"
    pretrained_models_output_path = "pretrained_models.zip"

    cacd_utkface_dataset_uid = rf"1--mrHW8QrbGSB92YDoWz5KgFXCfeF2GP"
    cacd_utkface_dataset_url = (
        rf"https://drive.google.com/uc?id={cacd_utkface_dataset_uid}"
    )
    cacd_utkface_dataset_output_path = "cacd_utkface.zip"

    # downloading from google drive
    gdown.download(pretrained_models_url, pretrained_models_output_path, quiet=False)
    gdown.download(
        cacd_utkface_dataset_url, cacd_utkface_dataset_output_path, quiet=False
    )
    # creating the dummy folders
    os.makedirs("cacd_utkface_hold_dir", exist_ok=True)
    os.makedirs("pretrained_models_hold_dir", exist_ok=True)

    out_dir1 = os.path.join(".", "pretrained_models_hold_dir")
    out_dir2 = os.path.join(".", "cacd_utkface_hold_dir")
    # unziping
    with zipfile.ZipFile(pretrained_models_output_path, "r") as zip_ref:
        zip_ref.extractall(out_dir1)

    with zipfile.ZipFile(cacd_utkface_dataset_output_path, "r") as zip_ref:
        zip_ref.extractall(out_dir2)

    pretrained_models_hold_halfpath = (
        "content/drive/MyDrive/ebrahimi_moghadam_refactored_final/criteria"
    )
    cacd_utkface_dataset_halfpath = "content/drive/MyDrive/data"

    pretrained_models_src_path = os.path.join(out_dir1, pretrained_models_hold_halfpath)
    cacd_utkface_src_path = os.path.join(out_dir2, cacd_utkface_dataset_halfpath)

    # #################################3
    #
    #  ONLY YOU HAVE TO CHANGE THE DEST PATHS
    #
    ###################################3

    pretrained_models_dest_apth = os.path.join(".", "criteria")
    cacd_utkface_dest_apth = "."

    # copt to the right place in the pipeline
    shutil.copytree(
        src=pretrained_models_src_path,
        dst=pretrained_models_dest_apth,
        dirs_exist_ok=True,
    )

    shutil.copytree(
        src=cacd_utkface_src_path, dst=cacd_utkface_dest_apth, dirs_exist_ok=True
    )


def get_from_storage_celeba_10000():
    def get_file(public_link, src_name, dest_name):
        nc = nextcloud_client.Client.from_public_link(public_link)
        nc.get_file(src_name, dest_name)
    # getting celeba alligned and cropped
    get_file(
        "https://storage.cse-sbu.ir/s/pRkpHd36YHN6agZ",
        src_name="/",
        dest_name="./celeba_aligne_cropped.zip",
    )
    # getting annotations celeba_10000
    get_file(
        "https://storage.cse-sbu.ir/s/eHtPE6K36JGRo2j",
        src_name="/",
        dest_name="./annotation_10000_celeba.csv",
    )
    out_dir1 = "."
    with zipfile.ZipFile("celeba_aligne_cropped.zip", "r") as zip_ref:
        zip_ref.extractall(out_dir1)


def get_from_storage_rage_gan_data():
    def get_file(public_link, src_name, dest_name):
        nc = nextcloud_client.Client.from_public_link(public_link)
        nc.get_file(src_name, dest_name)
    # getting rage_gan alligned and cropped
    get_file(
        "https://storage.cse-sbu.ir/s/cTzeHwGgiM63nZe",
        src_name="/",
        dest_name="./Rage_gan_cropped_restored_dataset.zip",
    )
    # getting annotations train
    get_file(
        "https://storage.cse-sbu.ir/s/FeEJarKySsMKXWo",
        src_name="/",
        dest_name="./Rage_gan_cropped_restored_train_dataset_annotation.csv",
    )
    # getting annotations test
    get_file(
        "https://storage.cse-sbu.ir/s/nxbt97dPmMKbCTE",
        src_name="/",
        dest_name="./Rage_gan_cropped_restored_test_dataset_annotation.csv",
    )
    out_dir1 = "."
    with zipfile.ZipFile("Rage_gan_cropped_restored_dataset.zip", "r") as zip_ref:
        zip_ref.extractall(out_dir1)

def get_from_storage_mark2():
    def get_file(public_link, src_name, dest_name):
        nc = nextcloud_client.Client.from_public_link(public_link)
        nc.get_file(src_name, dest_name)
    # get data from storage
    get_file(
        "https://storage.cse-sbu.ir/s/dDmz5RLS93fGHn3",
        src_name="/",
        dest_name="./mark2_pretrained_models.zip",
    )
    get_file(
        "https://storage.cse-sbu.ir/s/NnF4L6EHYPqJtE8",
        src_name="/",
        dest_name="./res34_fair_align_multi_7.pt",
    )
    get_file(
        "https://storage.cse-sbu.ir/s/5eabWi8sypr9Lq3",
        src_name="/",
        dest_name="./alex.pth",
    )
    get_file(
        "https://storage.cse-sbu.ir/s/yoxAWnkeqGsraJB",
        src_name="/",
        dest_name="./alexnet-owt-7be5be79.pth",
    )
    out_dir1 = "."
    with zipfile.ZipFile("mark2_pretrained_models.zip", "r") as zip_ref:
        zip_ref.extractall(out_dir1)

def get_from_storage_artifacts():
    def get_file(public_link, src_name, dest_name):
        nc = nextcloud_client.Client.from_public_link(public_link)
        nc.get_file(src_name, dest_name)
    # getting celeba alligned and cropped
    get_file(
        "https://storage.cse-sbu.ir/s/Yqr7BiQyqsbrwm3",
        src_name="/",
        dest_name="./artifacts_10.zip",
    )
    # getting pretrained models
    out_dir1 = "."
    with zipfile.ZipFile("artifacts_10.zip", "r") as zip_ref:
        zip_ref.extractall(out_dir1)


def get_from_storage_kinface_data():
    def get_file(public_link, src_name, dest_name):
        nc = nextcloud_client.Client.from_public_link(public_link)
        nc.get_file(src_name, dest_name)
    # getting rage_gan final weights
    # get_file(
    #     "https://storage.cse-sbu.ir/s/y85Yt2GWHxXrHem",
    #     src_name="/",
    #     dest_name="./RAGEGAN_weights_final.zip",
    # )
    get_file(
        "https://storage.cse-sbu.ir/s/tkXyf5kpaCWDmJR",
        src_name="/",
        dest_name="./RAGEGAN_better_10_cyclic_epoch_weights.zip",
    )
    # getting kinface1
    get_file(
        "https://storage.cse-sbu.ir/s/mDymrYyNo6g9RSk",
        src_name="/",
        dest_name="./Kinface1_mirror_augment_high_quality.zip",
    )
    # getting kinface2
    get_file(
        "https://storage.cse-sbu.ir/s/dcXAsJBALkPnxwR",
        src_name="/",
        dest_name="./Kinface2_mirror_augment_high_quality.zip",
    )
    # getting  kinface full face
    get_file(
        "https://storage.cse-sbu.ir/s/HBgTqHPWFkqF3KJ",
        src_name="/",
        dest_name="./KinFace1_FULL_FACE.zip",
    )
    out_dir1 = "."
    os.makedirs("hold_pretrained_Ragegan")
    with zipfile.ZipFile("RAGEGAN_better_10_cyclic_epoch_weights.zip", "r") as zip_ref:
        zip_ref.extractall("hold_pretrained_Ragegan")
    out_dir1 = "."
    with zipfile.ZipFile("Kinface1_mirror_augment_high_quality.zip", "r") as zip_ref:
        zip_ref.extractall(out_dir1)
    out_dir1 = "."
    with zipfile.ZipFile("Kinface2_mirror_augment_high_quality.zip", "r") as zip_ref:
        zip_ref.extractall(out_dir1)
    out_dir1 = "."
    with zipfile.ZipFile("KinFace1_FULL_FACE.zip", "r") as zip_ref:
        zip_ref.extractall(out_dir1)

def test_Rage_kinfaces(net):
    img_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # get_from_storage_kinface_data()

    # # Loading models pretrained weights
    # path_feature_mixer = "hold_pretrained_Ragegan/artifacts/epoch50/feature_mixer.pt"
    # path_pretrained_encoder = "hold_pretrained_Ragegan/artifacts/epoch50/pretrained_encoder.pt"
    # path_race_mixer1 = "hold_pretrained_Ragegan/artifacts/epoch50/race_mixer1.pt"
    # path_race_mixer2 = "hold_pretrained_Ragegan/artifacts/epoch50/race_mixer2.pt"
    # path_race_mixer3 = "hold_pretrained_Ragegan/artifacts/epoch50/race_mixer3.pt"

    # ckpt_feature_mixer= torch.load(path_feature_mixer,map_location="cpu")
    # ckpt_pretrained_encoder= torch.load(path_pretrained_encoder,map_location="cpu")
    # ckpt_race_mixer1= torch.load(path_race_mixer1,map_location="cpu")
    # ckpt_race_mixer2= torch.load(path_race_mixer2,map_location="cpu")
    # ckpt_race_mixer3= torch.load(path_race_mixer3,map_location="cpu")

    # net.feature_mixer.load_state_dict(ckpt_feature_mixer)
    # net.pretrained_encoder.load_state_dict(ckpt_pretrained_encoder)
    # net.race_mixers[0].load_state_dict(ckpt_race_mixer1)
    # net.race_mixers[1].load_state_dict(ckpt_race_mixer2)
    # net.race_mixers[2].load_state_dict(ckpt_race_mixer3)


    #eval setting the model
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    ages = [20,30,40,50,60,70,80]
    # kinface1_folder ="Kinface1_mirror_augment_high_quality"
    kinface1_folder ="KinFace1_FULL_FACE"
    kinface2_folder ="Kinface2_mirror_augment_high_quality"
    os.makedirs("Kinface_Ragegan_outputs/kinface1",exist_ok=True)
    os.makedirs("Kinface_Ragegan_outputs/kinface2",exist_ok=True)
    sub_dirs = ["father-dau","father-son" ,"mother-dau" , "mother-son"]

    # kinface1 outputs
    for sub_dir in sub_dirs:
        input_sub_dir = os.path.join(kinface1_folder,sub_dir)
        output_sub_dir = os.path.join("Kinface_Ragegan_outputs/kinface1",sub_dir)
        for age in ages:
            print(f"on {age=} :")
            out_subdir =os.path.join(output_sub_dir, str(age)) 
            os.makedirs(out_subdir , exist_ok=True)
            # for index in range(1):
            for file_name in os.listdir(input_sub_dir):
                # try:
                image = Image.open(os.path.join(input_sub_dir,file_name))
                image_tensor = img_transforms(image)
                image_with_age = torch.cat((image_tensor , (age/100)*torch.ones(1,256,256))).unsqueeze(dim=0)
                out_image_tensor = net(image_with_age.to("cuda") ,resize=True)
                # print(type(out_image_tensor))
                # print(f"{out_image_tensor.shape=}")
                out_image = tensor2im(out_image_tensor.squeeze(dim=0).detach().cpu())
                out_image.save(os.path.join(out_subdir , file_name))
                # except:
                #     continue
            print(len(os.listdir(out_subdir)))

    # kinface2 outputs
    for sub_dir in sub_dirs:
        input_sub_dir = os.path.join(kinface2_folder,sub_dir)
        output_sub_dir = os.path.join("Kinface_Ragegan_outputs/kinface2",sub_dir)
        for age in ages:
            print(f"on {age=} :")
            out_subdir =os.path.join(output_sub_dir, str(age)) 
            os.makedirs(out_subdir , exist_ok=True)
            # for index in range(1):
            for file_name in os.listdir(input_sub_dir):
                # try:
                image = Image.open(os.path.join(input_sub_dir,file_name))
                image_tensor = img_transforms(image)
                image_with_age = torch.cat((image_tensor , (age/100)*torch.ones(1,256,256))).unsqueeze(dim=0)
                out_image_tensor = net(image_with_age.to("cuda") ,resize=True)
                # print(type(out_image_tensor))
                # print(f"{out_image_tensor.shape=}")
                out_image = tensor2im(out_image_tensor.squeeze(dim=0).detach().cpu())
                out_image.save(os.path.join(out_subdir , file_name))
                # except:
                #     continue
            print(len(os.listdir(out_subdir)))
    # uploading the files
    zf = zipfile.ZipFile("Kinface_Ragegan_outputs.zip", "w")
    for dirname, subdirs, files in os.walk("Kinface_Ragegan_outputs"):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()

    nc = nextcloud_client.Client('https://storage.cse-sbu.ir')

    nc.login('GitlabOID-69', 'M0uCoqH+wjvTYC0G8trh9jGYbxdecRcFhXXxk9WGPoE=')

    nc.put_file('testdir/Kinface_Ragegan_better_outputs.zip', 'Kinface_Ragegan_outputs.zip')
    print("Done !")

def test_SAM_kinfaces(net):
    img_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    get_from_storage_kinface_data()

    #eval setting the model
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    ages = [20,30,40,50,60,70,80]
    kinface1_folder ="Kinface1_mirror_augment_high_quality"
    kinface2_folder ="Kinface2_mirror_augment_high_quality"
    os.makedirs("Kinface_SAM_outputs/kinface1",exist_ok=True)
    os.makedirs("Kinface_SAM_outputs/kinface2",exist_ok=True)
    sub_dirs = ["father-dau","father-son" ,"mother-dau" , "mother-son"]

    # kinface1 outputs
    for sub_dir in sub_dirs:
        input_sub_dir = os.path.join(kinface1_folder,sub_dir)
        output_sub_dir = os.path.join("Kinface_SAM_outputs/kinface1",sub_dir)
        for age in ages:
            print(f"on {age=} :")
            out_subdir =os.path.join(output_sub_dir, str(age)) 
            os.makedirs(out_subdir , exist_ok=True)
            # for index in range(1):
            for file_name in os.listdir(input_sub_dir):
                # try:
                image = Image.open(os.path.join(input_sub_dir,file_name))
                image_tensor = img_transforms(image)
                image_with_age = torch.cat((image_tensor , (age/100)*torch.ones(1,256,256))).unsqueeze(dim=0)
                out_image_tensor = net(image_with_age.to("cuda") ,resize=True)
                # print(type(out_image_tensor))
                # print(f"{out_image_tensor.shape=}")
                out_image = tensor2im(out_image_tensor.squeeze(dim=0).detach().cpu())
                out_image.save(os.path.join(out_subdir , file_name))
                # except:
                #     continue
            print(len(os.listdir(out_subdir)))

    # kinface2 outputs
    for sub_dir in sub_dirs:
        input_sub_dir = os.path.join(kinface2_folder,sub_dir)
        output_sub_dir = os.path.join("Kinface_SAM_outputs/kinface2",sub_dir)
        for age in ages:
            print(f"on {age=} :")
            out_subdir =os.path.join(output_sub_dir, str(age)) 
            os.makedirs(out_subdir , exist_ok=True)
            # for index in range(1):
            for file_name in os.listdir(input_sub_dir):
                # try:
                image = Image.open(os.path.join(input_sub_dir,file_name))
                image_tensor = img_transforms(image)
                image_with_age = torch.cat((image_tensor , (age/100)*torch.ones(1,256,256))).unsqueeze(dim=0)
                out_image_tensor = net(image_with_age.to("cuda") ,resize=True)
                # print(type(out_image_tensor))
                # print(f"{out_image_tensor.shape=}")
                out_image = tensor2im(out_image_tensor.squeeze(dim=0).detach().cpu())
                out_image.save(os.path.join(out_subdir , file_name))
                # except:
                #     continue
            print(len(os.listdir(out_subdir)))
    # uploading the files
    zf = zipfile.ZipFile("Kinface_SAM_outputs.zip", "w")
    for dirname, subdirs, files in os.walk("Kinface_SAM_outputs"):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()

    nc = nextcloud_client.Client('https://storage.cse-sbu.ir')

    nc.login('GitlabOID-69', 'M0uCoqH+wjvTYC0G8trh9jGYbxdecRcFhXXxk9WGPoE=')

    nc.put_file('testdir/Kinface_SAM_outputs.zip', 'Kinface_SAM_outputs.zip')
    print("Done !")

if __name__ == "__main__":
    # keyboared = Keyboared()
    # args = keyboared.parser.parse_args()
    # with open("config.json", "w") as fp:
    #     json.dump(vars(args), fp)

    get_from_storage_rage_gan_data()
    get_from_storage_mark2()

    get_from_storage_kinface_data()

    with open("config.json", "r") as fp:
        opts = json.load(fp)
    print(opts)
    opts = argparse.Namespace(**opts)
    print(opts.gender)
    args = opts
    consts.NUM_Z_CHANNELS = args.z_channels
    network = network.Net(opts=args)
    opts = args

    if not args.cpu and torch.cuda.is_available():
        network.cuda()

    if args.mode == "train":

        betas = (args.b1, args.b2) if args.load is None else None
        weight_decay = args.weight_decay if args.load is None else None
        lr = args.learning_rate if args.load is None else None

        if args.load is not None:
            network.load(args.load)
            print("Loading pre-trained models from {}".format(args.load))

        data_src = args.input or consts.UTKFACE_DEFAULT_PATH
        print("Data folder is {}".format(data_src))
        results_dest = args.output or default_train_results_dir()
        os.makedirs(results_dest, exist_ok=True)
        print("Results folder is {}".format(results_dest))

        with open(
            os.path.join(results_dest, "session_arguments.txt"), "w"
        ) as info_file:
            info_file.write(" ".join(sys.argv))

        log_path = os.path.join(results_dest, "log_results.log")
        if os.path.exists(log_path):
            os.remove(log_path)
        logging.basicConfig(filename=log_path, level=logging.DEBUG)

        # Loading models pretrained weights
        # path_feature_mixer = "hold_pretrained_Ragegan/artifacts/epoch50/feature_mixer.pt"
        # path_pretrained_encoder = "hold_pretrained_Ragegan/artifacts/epoch50/pretrained_encoder.pt"
        # path_race_mixer1 = "hold_pretrained_Ragegan/artifacts/epoch50/race_mixer1.pt"
        # path_race_mixer2 = "hold_pretrained_Ragegan/artifacts/epoch50/race_mixer2.pt"
        # path_race_mixer3 = "hold_pretrained_Ragegan/artifacts/epoch50/race_mixer3.pt"
        # path_feature_mixer = "hold_pretrained_Ragegan/artifacts/epoch10/feature_mixer.pt"
        # path_pretrained_encoder = "hold_pretrained_Ragegan/artifacts/epoch10/pretrained_encoder.pt"
        # path_race_mixer1 = "hold_pretrained_Ragegan/artifacts/epoch10/race_mixer1.pt"
        # path_race_mixer2 = "hold_pretrained_Ragegan/artifacts/epoch10/race_mixer2.pt"
        # path_race_mixer3 = "hold_pretrained_Ragegan/artifacts/epoch10/race_mixer3.pt"

        # ckpt_feature_mixer= torch.load(path_feature_mixer,map_location="cpu")
        # ckpt_pretrained_encoder= torch.load(path_pretrained_encoder,map_location="cpu")
        # ckpt_race_mixer1= torch.load(path_race_mixer1,map_location="cpu")
        # ckpt_race_mixer2= torch.load(path_race_mixer2,map_location="cpu")
        # ckpt_race_mixer3= torch.load(path_race_mixer3,map_location="cpu")

        # network.net.feature_mixer.load_state_dict(ckpt_feature_mixer)
        # network.net.pretrained_encoder.load_state_dict(ckpt_pretrained_encoder)
        # network.net.race_mixers[0].load_state_dict(ckpt_race_mixer1)
        # network.net.race_mixers[1].load_state_dict(ckpt_race_mixer2)
        # network.net.race_mixers[2].load_state_dict(ckpt_race_mixer3)

        network.teach(
            utkface_path=data_src,
            batch_size=args.batch_size,
            betas=betas,
            epochs=args.epochs,
            weight_decay=weight_decay,
            lr=lr,
            should_plot=args.sp,
            where_to_save=results_dest,
            models_saving=args.models_saving,
        )

        test_single(network.net, "RAGEGAN_better_final_race_outputs" , "rage_gan_test_annotations","rage_gan_test")
        test_Rage_kinfaces(network.net)
        # test_SAM_kinfaces(network.net)
        
        # saving the weights
        zf = zipfile.ZipFile("artifacts.zip", "w")
        for dirname, subdirs, files in os.walk("artifacts"):
            zf.write(dirname)
            for filename in files:
                zf.write(os.path.join(dirname, filename))
        zf.close()

        nc = nextcloud_client.Client('https://storage.cse-sbu.ir')

        nc.login('GitlabOID-69', 'M0uCoqH+wjvTYC0G8trh9jGYbxdecRcFhXXxk9WGPoE=')

        # nc.mkdir('testdir')

        nc.put_file('testdir/RAGEGAN_better_20_cyclic_epoch_weights.zip', 'artifacts.zip')
        print("Done !")
        
        # # saving the generated files
        zf = zipfile.ZipFile("RAGEGAN_better_final_race_outputs.zip", "w")
        for dirname, subdirs, files in os.walk("RAGEGAN_better_final_race_outputs"):
            zf.write(dirname)
            for filename in files:
                zf.write(os.path.join(dirname, filename))
        zf.close()

        nc = nextcloud_client.Client('https://storage.cse-sbu.ir')

        nc.login('GitlabOID-69', 'M0uCoqH+wjvTYC0G8trh9jGYbxdecRcFhXXxk9WGPoE=')

        # nc.mkdir('testdir')

        nc.put_file('testdir/RAGEGAN_better_final_race_outputs.zip', 'RAGEGAN_better_final_race_outputs.zip')
        print("Done !")


    elif args.mode == "test":

        # if args.load is None:
        #     raise RuntimeError("Must provide path of trained models")

        # network.load(path=args.load, slim=True)
        network.eval()
        images = torch.ones((1,3,256,256))
        ages = torch.tensor([int(30)]).unsqueeze(dim=0)
        genders = 1
        if genders == "Woman":
            genders = 0
        else:
            genders = 1
        genders = torch.tensor([genders]).unsqueeze(dim=1)
        print(f"{images}")
        out , _ = network.test(
            images = images,
            ages = ages,
            genders = genders,
            no_aging="not permitted"
        )
        print(f"{out.shape=}")
        from  torchvision.transforms import transforms
        out = out.cpu().detach().squeeze(dim=0)  # Remove batch dimension
        print(f"{out.shape=}")
        to_pil = transforms.ToPILImage()  # Create the ToPILImage transform
        out_pil = to_pil(out)  # Convert tensor to PIL image
        out_pil.save("out.jpg")