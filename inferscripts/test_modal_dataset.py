import os
import json
from tqdm import tqdm 


# 视频理解
if False:
    # 读取 JSON 文件
    file_path = '/storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/uni_DenseFusion-4V-100k.json'  # 替换为你的 JSON 文件路径
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化最长长度
    max_length = 0

    # 遍历 JSON 数据中的每个对话
    for item in tqdm(data):
        for conversation in item['conversations']:
            text_length = len(conversation['value'])
            
            if text_length > max_length:
                max_length = text_length
            print(f'Max length: {max_length}, {text_length=}')

    # 打印最长长度
    print(f"The longest conversation text length is: {max_length}")

# 文本理解
if True:
    # 读取 JSON 文件
    file_path = '/storage/yqs/dataset/BAAI/Infinity-Instruct/uni_Gen.json'  # 替换为你的 JSON 文件路径
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化最长长度
    max_length = 0

    # 遍历 JSON 数据中的每个对话
    for item in tqdm(data):
        for conversation in item['conversations']:
            text_length = len(conversation['value'])
            
            if text_length > max_length:
                max_length = text_length
            print(f'Max length: {max_length}, {text_length=}')

    # 打印最长长度
    print(f"The longest conversation text length is: {max_length}")

"""
cd /storage/zhubin/Janus-MoE/
python test_modal_dataset.py

"""