import numpy as np
import json
from torch.utils.data import Dataset 
import random, os, json

# 定义一个简单的数据集
class ImageGenDataset(Dataset):
    def __init__(self,  
                image_gen_data_files=None,
                image_gen_rootdir=None, 
                ):
        
        # specific data file
        self.image_gen_data_files =  image_gen_data_files 
        self.image_gen_rootdir = image_gen_rootdir
        # load meta data
        # self.image_gen_data = self.read_jsonfile(self.image_gen_data_file); print(f'image_gen_data:{len(self.image_gen_data)}!!!')
        self.image_gen_data = []
        for data_file in self.image_gen_data_files:
            data = self.read_jsonfile(data_file)
            self.image_gen_data.extend(data)
        print(f'image_gen_data:{len(self.image_gen_data)}!!!')
  
    def __len__(self):
        return len(self.image_gen_data)
    
    def read_jsonfile(self, jsonfile):
        with open(jsonfile, 'r', encoding='utf-8') as f:
            return json.load(f)
    def __getitem__(self, idx):
        idx = random.randint(0, len(self.image_gen_data) - 1)

        data_item = self.image_gen_data[idx]
        if 'image_gen' in data_item:
            data_item['image_gen'] = [os.path.join(self.image_gen_rootdir, data_item['image_gen'][0])]
        return data_item
        
        """
        for key, value in *data.items():
            print(f"{key}: {value}")
        """