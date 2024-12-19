import network
import json
import consts
import logging
import os
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

import gdown
import zipfile
import os
import shutil


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


if __name__ == "__main__":
    keyboared = Keyboared()
    args = keyboared.parser.parse_args()
    with open("config.json", "w") as fp:
        json.dump(vars(args), fp)

    # with open("config.json", "r") as fp:
    #     opts = json.load(fp)
    # print(opts)
    # opts = argparse.Namespace(**opts)
    # print(opts.gender)
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

    elif args.mode == "test":

        if args.load is None:
            raise RuntimeError("Must provide path of trained models")

        network.load(path=args.load, slim=True)

        results_dest = args.output or default_test_results_dir()
        if not os.path.isdir(results_dest):
            os.makedirs(results_dest)

        image_tensor = pil_to_model_tensor_transform(pil_loader(args.input)).to(
            network.device
        )
        network.test_single(
            image_tensor=image_tensor,
            age=args.age,
            gender=args.gender,
            target=results_dest,
            watermark=args.watermark,
        )
