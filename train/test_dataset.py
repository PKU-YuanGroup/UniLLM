import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import gc

 


'''
The annotation files are consist of a list of dictionaries, where each item follows the following format:
[
    {# 图片理解数据格式
        "image": ["images/xxx.jpg"],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat are the colors of the bus in the image?"
            },
            {
                "from": "gpt",
                "value": "The bus in the image is white and red."
            },
            ...
        ]
    },
    {# 视频理解数据格式
        "video": ["videos/xxx.mp4"],
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nWhat are the main activities that take place in the video?"
            },
            {
                "from": "gpt",
                "value": "The main activities that take place in the video are the preparation of camera equipment by a man, a group of men riding a helicopter, and a man sailing a boat through the water."
            },
            ...
        ]
    },
    {# 纯文本数据格式
        "conversations": [
            {
                "from": "human",
                "value": "What are the main activities that take place in the video?"
            },
            {
                "from": "gpt",
                "value": "The main activities that take place in the video are the preparation of camera equipment by a man, a group of men riding a helicopter, and a man sailing a boat through the water."
            },
            ...
        ]
    },
    {# 图片生成格式
        "image_gen": ["images/xxx.jpg"],
        "conversations": [
            {
                "from": "human",
                "value": ["long caption", "short caption"]
            },
            {
                "from": "gpt",
                "value": ""
            },
            ...
        ]
    },
    ...
]

'''

"""
class SimpleDistributedSampler(Sampler):
    def __init__(self, data_source, num_replicas=None, rank=None, shuffle=False, 
                epoch=1, video_sampler_batchsize=2, image_sampler_batchsize=3, video_data_step_ratio=1/4,
                ):
        
        self.data_source = data_source  
        # print(self.data_source)
        self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()
        self.shuffle = shuffle
    
        # 每个进程负责的样本数
        self.num_samples = math.ceil(len(self.data_source) / self.num_replicas)
        
        # 数据总量对齐到进程数的倍数
        self.total_size = self.num_samples * self.num_replicas


        self.epoch = epoch
        self.video_data_step_ratio = video_data_step_ratio # 视频训练iteration占总iteration的比例
        self.video_sampler_batchsize = video_sampler_batchsize # getitem一次返回一个视频idx列表，列表长度为video_sampler_batchsize
        self.image_sampler_batchsize = image_sampler_batchsize # getitem一次返回一个图像idx列表，列表长度为image_sampler_batchsize

        self.pad_data_source_len =  image_sampler_batchsize * (1/video_data_step_ratio-1)  / video_sampler_batchsize * len(self.data_source) 
        self.pad_data_source_len = int(self.pad_data_source_len)

    def __iter__(self):

        # 创建索引
        indices = list(range(self.pad_data_source_len))
        # 如果需要 shuffle，则每个 epoch 打乱顺序
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)  # 使用epoch 作为种子
            indices = torch.randperm(self.pad_data_source_len, generator=g).tolist()

        # 扩展到可以整除 num_replicas 的长度
        indices += indices[:(self.total_size - len(indices))]

        # 视频batch和图像batch完成一次交替需要的idx个数，与sampler_batchsize, video_data_step_ratio, image_scale_bs都有关
        # idx_nums_of_data_type_change = self.video_sampler_batchsize * 1 + self.image_sampler_batchsize * (1/self.video_data_step_ratio - 1)
        # idx_nums_of_data_type_change = int(idx_nums_of_data_type_change)

        # 每个gpu可以分到的indices个数, 一个idx本来是一个数字，现在变成了一个列表
        indices = indices[self.rank:self.total_size:self.num_replicas]
        len_of_indices = int(len(indices) // self.video_sampler_batchsize)
        
        new_indices= []
        for i in range(len_of_indices):
            # video_idx   第一个是视频idx_list, 长度为video_sampler_batchsize
            s_videoidx1 = i * self.video_sampler_batchsize
            e_videoidx1 = (i+1) * self.video_sampler_batchsize
            item = indices[s_videoidx1:e_videoidx1] 
            item.append('video')
            new_indices.append(item)

            # image_idx  if self.video_data_step_ratio==4 后面3个是图像idx_list， 每一个长度为image_sampler_batchsize
            for i in range(int(1/self.video_data_step_ratio)-1):
                s_imageidx = i * self.image_sampler_batchsize 
                e_imageidx = (i+1) * self.image_sampler_batchsize

                item = indices[s_imageidx:e_imageidx] 
                item.append('image')
                new_indices.append(item)

        return iter(new_indices)

    def __len__(self):
        return  self.num_samples

    def set_epoch(self, epoch):
        # 在分布式训练时更新 epoch，从而保证每个 epoch 的 shuffle 不同
        self.epoch = epoch"""

from train.dataset_files.image_gen_dataset import ImageGenDataset
from train.dataset_files.image_under_dataset import ImageUnderDataset
from train.dataset_files.text_chat_dataset  import TextChatDataset

# import sys; sys.path.append('/storage/jp/Janus')
from janus.utils.io import load_pil_images
from janus.models import MultiModalityCausalLM, VLChatProcessor

import numpy as np
import json, os
import traceback
import random
import warnings
import copy
from dataclasses import dataclass, field
import transformers
from typing import Dict, List, Optional, Sequence
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
                image_file = [os.path.join(data_folder, f) for f in image_file]
            else:
                image_file = [os.path.join(data_folder, image_file)]
        elif 'image_gen' in data_dict and data_dict['image_gen'] is not None:
            modal = 'image_gen'
            image_file = data_dict['image_gen']
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
        for _ in range(batchsize):
            messages = data[0]
            # assert self.image_gen_rootdir == self.image_under_rootdir 
            data_folder = self.all_datasets_rootdir[ind]

            try:
                modal, messages = self._convert_normal(
                                        messages,
                                        data_folder=data_folder, 
                                        short_cap=self.short_cap
                                        )
                # load images and prepare for inputs 
                pil_images = load_pil_images(messages)
                # data_dict = self.vlprocessor(
                #         conversations=messages, images=pil_images, force_batchify=True, is_training=True, modal=modal)
                
                # import ipdb; ipdb.set_trace()
                data_dict =  self.vlprocessor.process_one(
                            conversations=messages, images=pil_images, is_training=True, modal=modal
                        )
                
                

            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when process {idx}-th example: {data_dict}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)
            
            # data_batch.append(dict(data_dict))
            data_batch.append( data_dict )
        
        # 
        data_batch =  self.vlprocessor.batchify(data_batch)
        
        data_batch = dict(data_batch)
        data_batch['modals'] = [modal]*batchsize
        return data_batch

        # return [data[0] for _ in range(batchsize)] 


@dataclass
class DataCollatorWithFlatteningForSupervisedDataset(object):
    """Collate examples for batch flattened supervised fine-tuning."""

    vlprocessor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict], separator_id=-100) -> Dict[str, torch.Tensor]:

        assert len(instances) == 1, 'batchsize必须是1，因为batchfy已经在getitem里面执行了！'
        batch = instances[0]
        return batch
        # instances = instances[0]
        """import ipdb; ipdb.set_trace()
        batch = dict()
        # work for 'images' argument in `prepare_inputs_labels_for_multimodal`
        # batch["sft_format"] = torch.cat([x["sft_format"] for x in instances])
        # try:
        #     batch["input_ids"] = torch.cat([x["input_ids"] for x in instances])
        # except Exception as e:
        #     import ipdb; ipdb.set_trace()
        #     print(e) 

        batch["input_ids"] = torch.cat([x["input_ids"] for x in instances])
        batch["labels"] = torch.cat([x["labels"] for x in instances])
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances])
        batch["attention_mask"] = torch.cat([x["attention_mask"] for x in instances])
        batch["images_seq_mask"] = torch.cat([x["images_seq_mask"] for x in instances])
        batch["images_emb_mask"] = torch.cat([x["images_emb_mask"] for x in instances])
        batch["modals"] = sum([x["modals"] for x in instances], [])
        return batch"""


model_path = "deepseek-ai/Janus-Pro-7B"
model_path = "/storage/jp/Janus/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, cache_dir = './cache_dir')
vlprocessor  = vl_chat_processor
vlprocessor.tokenizer_model_max_length = 2048

tokenizer = vl_chat_processor.tokenizer

# 创建数据集实例
dataset = UniDataset(
    vlprocessor=vl_chat_processor,


    # image_under_data_files=['/storage/jp/Janus/trainer/dataset_files/image_under.json'], 
    # image_under_rootdir='/storage/jp/Janus',
    # image_gen_data_files=['/storage/jp/Janus/trainer/dataset_files/image_gen.json'],
    # image_gen_rootdir='/storage/jp/Janus',
    # text_chat_data_files=['/storage/jp/Janus/trainer/dataset_files/text_chat.json'],
    # samples_per_epoch=10000,
    # dataset="image_under||image_gen||text_chat",
    # sample_rate=[9, 3, 3],
    # batchsize_list=[1,2,3]

    image_under_data_files=['/storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/mini_uni_DenseFusion-4V-100k.json'], 
    image_under_rootdir='/storage/yqs/dataset/BAAI/DenseFusion-1M/images',
    image_gen_data_files=['/storage/dataset/filter_aes/cap_merge_final_640/recap2/mini_janus_part0_cap6595998.json'],
    image_gen_rootdir='/storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded',
    text_chat_data_files=['/storage/yqs/dataset/BAAI/Infinity-Instruct/mini_uni_Gen.json'],
    samples_per_epoch=1000000,
    dataset="image_under||image_gen||text_chat",
    sample_rate=[0, 1, 0],
    batchsize_list=[1,2,3]
)


"""
train_dataset = UniDataset(
        vlprocessor=vlprocessor,
        image_under_data_files=['/storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/mini_uni_DenseFusion-4V-100k.json'], 
        image_under_rootdir='/storage/yqs/dataset/BAAI/DenseFusion-1M/images',
        image_gen_data_files=['/storage/dataset/filter_aes/cap_merge_final_640/recap2/mini_janus_part0_cap6595998.json'],
        image_gen_rootdir='/storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded',
        text_chat_data_files=['/storage/yqs/dataset/BAAI/Infinity-Instruct/7M_domains/subjective/mini_output.json'],
        samples_per_epoch=1000000,
        dataset="image_under||image_gen||text_chat",
        sample_rate=[3, 3, 3],
        batchsize_list=[1,2,3]

    )

"""

ddp_flag = False

if ddp_flag:
    # 初始化进程组
    dist.init_process_group(backend="nccl")
    
    # 获取进程 rank 和进程总数
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 创建自定义分布式 Sampler
    sampler = SimpleDistributedSampler(dataset, num_replicas=world_size, rank=rank)

    # 使用 DataLoader 加载数据，传入自定义的分布式 Sampler
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, drop_last=True, num_workers=0)

    # 迭代并打印每个进程处理的 batch 数据
    for batch in dataloader:
        if rank == 0:
            import ipdb; ipdb.set_trace()
        print(f"Rank {rank}, Batch: {batch}")

    # 记得在最后销毁进程组
    dist.destroy_process_group()
else:
    dataloader = DataLoader(dataset,  batch_size=1, drop_last=True, num_workers=0, collate_fn=DataCollatorWithFlatteningForSupervisedDataset(vlprocessor=vl_chat_processor))
    # 迭代并打印每个进程处理的 batch 数据
    for batch in dataloader:
        import ipdb; ipdb.set_trace()
        # try:
        print(f"Batch: {len(batch)}, {batch['sft_format']}")


"""

# 单进程

cd /storage/jp/Janus
source /storage/miniconda3/etc/profile.d/conda.sh
conda activate janus_pro

cd /storage/zhubin/Janus-zb
python  test_dataset.py 


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