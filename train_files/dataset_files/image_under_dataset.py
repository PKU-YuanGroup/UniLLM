import numpy as np
import json
from torch.utils.data import Dataset 
import random, os, json

# 定义一个简单的数据集
class ImageUnderDataset(Dataset):
    def __init__(self,  
                image_under_data_files=None, 
                image_under_rootdir=None
                ):
        
        # specific data file
        self.image_under_data_files = image_under_data_files 

        # load meta data
        self.image_under_data = []
        for data_file in image_under_data_files:
            data = self.read_jsonfile(data_file)
            self.image_under_data.extend(data)
        print(f'image_under_data:{len(self.image_under_data)}!!!')
  
    def __len__(self):
        return len(self.image_under_data)
    
    def read_jsonfile(self, jsonfile):
        with open(jsonfile, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def __getitem__(self, idx):
        idx = random.randint(0, len(self.image_under_data) - 1)

        data_item = self.image_under_data[idx]
        if 'image_under' in data_item:
            data_item['image_under'] = [os.path.join(self.image_under_rootdir, data_item['image_under'][0])]
        return self.image_under_data[idx]
        