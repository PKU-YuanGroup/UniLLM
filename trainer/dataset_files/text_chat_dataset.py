import numpy as np
import json
from torch.utils.data import Dataset 
import random, os, json

# 定义一个简单的数据集
class TextChatDataset(Dataset):
    def __init__(self,  
                text_chat_data_files=None, 
              
                ):
        
        # specific data file
        self.text_chat_data_files = text_chat_data_files
        # self.text_chat_data = self.read_jsonfile(self.text_chat_data_file); print(f'text_chat_data:{len(self.text_chat_data)}!!!')
        # load meta data
        self.text_chat_data = []
        for data_file in self.text_chat_data_files:
            data = self.read_jsonfile(data_file)
            self.text_chat_data.extend(data)
        print(f'text_chat_data:{len(self.text_chat_data)}!!!')


    def read_jsonfile(self, jsonfile):
        with open(jsonfile, 'r', encoding='utf-8') as f:
            return json.load(f)
    def __len__(self):
        return len(self.text_chat_data)
    
    def __getitem__(self, idx):
        idx = random.randint(0, len(self.text_chat_data) - 1)
        return self.text_chat_data[idx]
        