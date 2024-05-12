import os
import json
import pdb
import sys
import numpy as np
import torch
from PIL import Image, ImageOps
import math
import glob
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self):
        self.source_list = glob.glob("../test/images/source/*.jpg")
        self.target_list = glob.glob("../test/images/target/*.jpg")
        # self.text_list = glob.glob("../test/text/*.txt")
        self.text_list = glob.glob("../../code/demo_output/*/text.txt")
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
    
class CustomDatasetHQ(Dataset):
    def __init__(self):
        self.source_list = glob.glob("../../code/demo_output_HQ/*/source.jpg")
        self.text_list = glob.glob("../../code/demo_output_HQ/*/text.txt")
        self.source_list.sort()
        self.text_list.sort()

    def __len__(self):
        return 100 # len(self.text_list)

    def __getitem__(self, i):
        with open(self.text_list[i], "r") as f:
            prompt = f.read().splitlines()[0]
        img_x = Image.open(self.source_list[i])
        width, height = img_x.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        img_x = ImageOps.fit(img_x, (width, height), method=Image.Resampling.LANCZOS)
        
        image_0 = np.array(img_x)
        image_0 = torch.FloatTensor(image_0)/255
        image_0 = image_0.permute(2, 0, 1)
        return image_0, prompt
    
class CustomDatasetHQ_EVAL(Dataset):
    def __init__(self):
        self.source_list = glob.glob("../../code/eval_output_HQ_5/*/source.jpg")
        self.target_list = glob.glob("../../code/eval_output_HQ_5/*/target.jpg")
        self.text_list = glob.glob("../../code/eval_output_HQ_5/*/text.txt")
        self.source_list.sort()
        self.target_list.sort()
        self.text_list.sort()

    def __len__(self):
        return 100 # len(self.text_list)

    def __getitem__(self, i):
        with open(self.text_list[i], "r") as f:
            prompt = f.read().splitlines()[0]
        img_x = Image.open(self.source_list[i])
        target = Image.open(self.target_list[i])
        width, height = img_x.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        img_x = ImageOps.fit(img_x, (width, height), method=Image.Resampling.LANCZOS)
        target = ImageOps.fit(target, (width, height), method=Image.Resampling.LANCZOS)
        
        image_0 = np.array(img_x)
        image_0 = torch.FloatTensor(image_0)/255
        image_0 = image_0.permute(2, 0, 1)
        
        target = np.array(target)
        return image_0, target, prompt