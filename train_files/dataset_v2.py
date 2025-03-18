import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import torch.distributed as dist
# from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import gc
 

import sys, os; cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cur_dir))
# import sys; sys.path.append('/storage/jp/Janus')
# import ipdb; ipdb.set_trace()

from dataset_files.image_gen_dataset import ImageGenDataset
from dataset_files.image_under_dataset import ImageUnderDataset
from dataset_files.text_chat_dataset  import TextChatDataset

from janus.utils.io import load_pil_images
from janus.models import MultiModalityCausalLM, VLChatProcessor

from typing import Dict, List, Optional, Sequence
import numpy as np
import json, os
import traceback
import random
import warnings
import copy
import transformers
from dataclasses import dataclass, field


# 定义一个简单的数据集
class UniDataset(Dataset):
    def __init__(self,  
                 
                vlprocessor=None,


                image_under_data_files=None, 
                image_under_rootdir=None,
                image_gen_data_files=None,
                image_gen_rootdir=None,
                text_chat_data_files=None,
                samples_per_epoch=10000,
                dataset="image_under||image_gen||text_chat",
                
                sample_rate=[9, 3, 3],
                batchsize_list=[1,2,3],

                short_cap=0.2,
                ):

        self.vlprocessor = vlprocessor

        self.samples_per_epoch = samples_per_epoch
        sample_rate = np.array(sample_rate)
        self.short_cap  = short_cap
        self.sample_rate = sample_rate / sample_rate.sum()
        self.batchsize_list = batchsize_list


        self.image_under_rootdir = image_under_rootdir
        self.image_gen_rootdir = image_gen_rootdir


        self.datasets = dataset.split("||")
        self.all_datasets = []
        self.all_datasets_rootdir = []


        for dataset in self.datasets:
            if dataset == "image_under":
                self.all_datasets.append(
                    ImageUnderDataset(
                       image_under_data_files,
                       image_under_rootdir
                    )
                )
                self.all_datasets_rootdir.append(self.image_under_rootdir)

            elif dataset == "image_gen":
                self.all_datasets.append(
                    ImageGenDataset(
                      image_gen_data_files,
                      image_gen_rootdir
                    )
                )
                self.all_datasets_rootdir.append(self.image_gen_rootdir)


            elif dataset == "text_chat":
                self.all_datasets.append(
                    TextChatDataset(
                        text_chat_data_files,
                       
                    )
                )
                self.all_datasets_rootdir.append('None')
             
    def __len__(self):

        return  self.samples_per_epoch

    def read_jsonfile(self, jsonfile):
        with open(jsonfile, 'r', encoding='utf-8') as f:
            return json.load(f)
        

    def _convert_normal(self, data_dict, data_folder=None, short_cap=0.2):
 
        conversation = copy.deepcopy(data_dict["conversations"])

        # data sanity check and repair
        start_idx = 0
        for sentence in conversation:
            if sentence["from"] == "human" or sentence["from"] == "system":
                break
            start_idx += 1
        if start_idx > 0:
            warnings.warn(f"Find {start_idx} non-user sentences at the beginning of the conversation, remove them automatically!")
            conversation = conversation[start_idx:]
        assert len(conversation) > 1, f"Invalid conversation"

        if 'image' in data_dict and data_dict['image'] is not None:
            modal = 'image'
            if all(not "<image>" in sentence["value"] for sentence in conversation):
                warnings.warn(f"Image tag not found in the conversation, add it automatically at the beginning!")
                conversation[0]["value"] = "<image_placeholder>\n" + conversation[0]["value"]
            image_file = data_dict['image']
            if isinstance(image_file, list):
                image_file = [os.path.join(self.image_under_rootdir, f) for f in image_file]
            else:
                image_file = [os.path.join(self.image_under_rootdir, image_file)]
        elif 'image_gen' in data_dict and data_dict['image_gen'] is not None:
            modal = 'image_gen'
            image_file = data_dict['image_gen']
            if isinstance(image_file, list):
                image_file = [os.path.join(self.image_gen_rootdir, f) for f in image_file]
            else:
                image_file = [os.path.join(self.image_gen_rootdir, image_file)]
            # print(image_file)
        else:
            modal = 'text'

        messages = []
        image_id = 0
        for conv in conversation:
            if conv["from"] == "human":
                if modal == 'image':
                    if "<image>" in conv["value"]:
                        messages.append({
                            "role": "<|User|>",
                            "content": conv["value"].replace("<image>", "<image_placeholder>"),
                            "images": [image_file[image_id]]
                        })
                        image_id += 1
                    else:
                        messages.append({
                            "role": "<|User|>",
                            "content": conv["value"].replace("<image>", "<image_placeholder>"),
                        })
                elif modal == 'image_gen':
                    if isinstance(conv["value"], list):
                        if len(conv["value"]) > 1 and random.random() <  short_cap:
                            prompt = conv["value"][1]
                        else:
                            prompt = conv["value"][0]
                    else:
                        prompt = conv["value"]
                    messages.append({
                        "role": "<|User|>",
                        "content": prompt,
                        "images": [image_file[image_id]]
                    })
                    image_id += 1
                else:
                    messages.append({
                        "role": "<|User|>",
                        "content": conv["value"],
                    })
            else:
                if modal == 'image_gen':
                    messages.append({
                        "role": "<|Assistant|>",
                        "content": self.vlprocessor.image_start_tag
                    })
                else:
                    messages.append({
                        "role": "<|Assistant|>",
                        "content": conv['value']
                    })

        return modal, messages

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        batchsize = self.batchsize_list[ind]
        data = self.all_datasets[ind]

        data_batch = []
        # import ipdb; ipdb.set_trace()
        for b_idx in range(batchsize):
            messages = data[b_idx]
            # assert self.image_gen_rootdir == self.image_under_rootdir 
            data_folder = self.image_gen_rootdir

            try:
                modal, messages = self._convert_normal(
                                        messages,
                                        data_folder=data_folder, 
                                        short_cap=self.short_cap
                                        )
                # load images and prepare for inputs 
                pil_images = load_pil_images(messages)
                # print(pil_images)
                # data_dict = self.vlprocessor(
                #         conversations=messages, images=pil_images, force_batchify=True, is_training=True, modal=modal)
                
                data_dict =  self.vlprocessor.process_one(
                            conversations=messages, images=pil_images, is_training=True, modal=modal
                        )
                
                
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.__len__()) - 1)
                print(f"Encounted error when process {idx}-th example: {data_dict}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)
            
            # data_batch.append(dict(data_dict))
            data_batch.append( data_dict )
        
        # 
        data_batch =  self.vlprocessor.batchify(data_batch)
        
        data_batch = dict(data_batch)
        data_batch['modals'] = [modal]*batchsize

        # print(data_batch['modals'], '###########')
        return data_batch

        # return [data[0] for _ in range(batchsize)] 




@dataclass
class DataCollatorWithFlatteningForSupervisedDataset(object):
    """Collate examples for batch flattened supervised fine-tuning."""

    vlprocessor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict], separator_id=-100) -> Dict[str, torch.Tensor]:

        # print(len(instances), '!!!!!!!!!!!!!!!!')
        # assert len(instances) == 1, 'batchsize必须是1，因为batchfy已经在getitem里面执行了！'
        batch = instances[0]
        return batch
    

"""

# 单进程
cd /storage/jp/Janus/
python test_dataset.py


# 多进程
cd  /storage/zhubin/UniLLM
nnodes=1
nproc_per_node=2
export master_addr=127.0.0.1
export master_port=29505
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate 


HF_DATASETS_OFFLINE=1 torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node  \
--master_addr=$master_addr --master_port=$master_port \
test_dataset.py  

"""