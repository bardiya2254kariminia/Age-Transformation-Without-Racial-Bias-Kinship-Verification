import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from configs.data_configs import DATASETS

from torchvision import transforms


class Celeba_10000_dataset(Dataset):
    def __init__(self):
        super(Celeba_10000_dataset).__init__()
        self.transform = DATASETS["rage_gan"]["transforms"]()
        self.transform = self.transform.get_transforms()["transform_image_length"]
        self.annotations_path = DATASETS["rage_gan"]["annotations"]
        self.image_folder = DATASETS["rage_gan"]["train_source_root"]
        self.annotations = pd.read_csv(
            self.annotations_path, index_col=False, skip_blank_lines=True
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # print(self.annotations.iloc[index].values)
        # file_path, age, gender = (
        #     self.annotations.iloc[index , 1],
        #     self.annotations.iloc[index , 2],
        #     self.annotations.iloc[index , 3],
        # )
        file_name = self.annotations.iloc[index , 1]
        # print(file_name)
        # print(file_path, age, gender)
        # image = Image.open("/content/download (3).jpg").convert("RGB")
        image = Image.open(os.path.join(self.image_folder, file_name)).convert("RGB")
        image = self.transform(image)
        age = file_name.split("_")[0]
        return image, torch.tensor(int(age)), torch.tensor(0)


if __name__ == "__main__":
    ds = Celeba_10000_dataset()
    print(ds.__getitem__(1))
