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

baseline_texts = [
    "Hot summer weather", 
    "Sad world of scented candles", 
    "Fairies in the mountains", 
    "Fun in the shadows", 
    "Fragrant breeze", 
    "Stinging Winter Light", 
    "Streetlights in a dark street", 
    "Neon-lit streets", 
    "A warm spring breeze", 
    "The light of cold ice", 
    "Hot campfire", 
    "A bonfire that has cooled down", 
    "Lovely candles", 
    "Colorful Stages", 
    "Green fluorescent light", 
    "Lights guarding the castle", 
    "Fireflies out of nowhere", 
    "Mysterious Warehouse", 
    "Delicious clouds in a dream", 
    "Shimmering pink clouds", 
    "Red light and green light dance", 
    "The sweet smell of bread", 
    "It's about to rain blue", 
    "Spooky light", 
    "A dim alleyway", 
    "Blue light", 
    "Pink light", 
    "Green light", 
    "Yellow light", 
    "Purple light", 
    "Cotton candy", 
    "Rainbow", 
    "Candy", 
    "Bread", 
    "Desert", 
    "Fire", 
    "Water", 
    "Computer", 
    "Neon", 
    "Horror", 
    "Scarry", 
    "Ghostly whispers", 
    "Twilight mist", 
    "Moonlit beach", 
    "Starry sky", 
    "Chilly autumn evening", 
    "Whispering woods", 
    "Thunderous applause", 
    "Silent mountains", 
    "Roaring river", 
]

class CustomDataset(Dataset):
    def __init__(self):
        self.source_list = glob.glob("../../code/demo_output_org_HQ/*/source.jpg")
        # self.target_list = glob.glob("../test/images/target/*.jpg")
        # self.text_list = glob.glob("../test/text/*.txt")
        self.text_list = glob.glob("../../code/demo_output_org_HQ/*/text.txt")
        self.source_list.sort()
        # self.target_list.sort()
        self.text_list.sort()

    def __len__(self):
        return 100 # len(self.text_list)

    def __getitem__(self, i):
        # with open(self.text_list[i], "r") as f:
        #     prompt = f.read().splitlines()[0]
        prompt = baseline_texts[i]
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
        
        # image_1 = Image.open(self.target_list[i])
        # image_1 = ImageOps.fit(image_1, (width, height), method=Image.Resampling.LANCZOS)
        # image_1 = np.array(image_1)
        # image_1 = torch.FloatTensor(image_1)/255
        # image_1 = image_1.permute(2, 0, 1)
        return image_0, prompt
    
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
    
class CustomDatasetHQ10x10_EVAL(Dataset):
    def __init__(self):
        self.source_list = glob.glob("../../code/eval_output_HQ10x10_5/*/source.jpg")
        self.target_list = glob.glob("../../code/eval_output_HQ10x10_5/*/target.jpg")
        self.text_list = glob.glob("../../code/eval_output_HQ10x10_5/*/text.txt")
        self.source_list.sort()
        self.target_list.sort()
        self.text_list.sort()

    def __len__(self):
        return 10 # len(self.text_list)

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
    
class CustomDataset_cfg(Dataset):
    def __init__(self):
        self.source_list = glob.glob("../../code/eval_output_cfg_5/*/source.jpg")
        self.target_list = glob.glob("../../code/eval_output_cfg_5/*/target.jpg")
        self.text_list = glob.glob("../../code/eval_output_cfg_5/*/text.txt")
        self.source_list.sort()
        self.target_list.sort()
        self.text_list.sort()

    def __len__(self):
        return 7 # len(self.text_list)

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
    
class CustomDatasetHQ_qualitative(Dataset):
    def __init__(self):
        self.source_list = glob.glob("../../qualitative/*.jpg")
        self.source_list.sort()

    def __len__(self):
        return 5

    def __getitem__(self, i):
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
        
        return image_0