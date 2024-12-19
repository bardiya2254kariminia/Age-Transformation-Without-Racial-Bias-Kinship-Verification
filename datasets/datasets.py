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
        self.transform = DATASETS["celeba_10000"]["transforms"]()
        self.transform = self.transform.get_transforms()["transform_image_length"]
        self.annotations_path = DATASETS["celeba_10000"]["annotations"]
        self.image_folder = DATASETS["celeba_10000"]["train_source_root"]
        self.annotations = pd.read_csv(
            self.annotations_path, index_col=False, skip_blank_lines=True
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # print(self.annotations.iloc[index].values)
        file_path, age, gender = (
            self.annotations.iloc[index][1],
            self.annotations.iloc[index][2],
            self.annotations.iloc[index][3],
        )
        print(file_path, age, gender)
        image = Image.open(os.path.join(self.image_folder, file_path))
        image = self.transform(image)
        if gender == "Woman":
            gender = 1
        else:
            gender = 0
        
        age_tensor = torch.zeros((100))
        gender_tensor = torch.zeros((2))
        # print(f"{age_tensor.shape = } , {gender_tensor.shape =}")
        age_tensor[int(age)]+=1
        gender_tensor[gender] +=1
        return image, age_tensor, gender_tensor


if __name__ == "__main__":
    ds = Celeba_10000_dataset()
    print(ds.__getitem__(1))
