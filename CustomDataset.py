import os
import json
import pdb
import sys
import numpy as np
import torch
from PIL import Image
import glob
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self):
        self.source_list = glob.glob("../test/images/source/*.jpg")
        self.target_list = glob.glob("../test/images/target/*.jpg")
        self.text_list = glob.glob("../test/text/*.txt")
        self.source_list.sort()
        self.target_list.sort()
        self.text_list.sort()

    def __len__(self):
        return 100 # len(self.text_list)

    def __getitem__(self, i):
        with open(self.text_list[i], "r") as f:
            prompt = f.read().splitlines()[0]
        image_0 = Image.open(self.source_list[i])
        image_1 = Image.open(self.target_list[i])
        image_0 = np.array(image_0.resize((512, 512)))
        image_1 = np.array(image_1.resize((512, 512)))
        image_0 = torch.FloatTensor(image_0)/255
        image_1 = torch.FloatTensor(image_1)/255
        image_0 = image_0.permute(2, 0, 1)
        image_1 = image_1.permute(2, 0, 1)
        return image_0, image_1, prompt