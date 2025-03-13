import copy
import json
import os
import random
import re
import sys
import warnings
import traceback
from packaging import version
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import os
# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
import transformers
from packaging import version
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from arg_util import ModelArguments, DataArguments, TrainingArguments
from janus.utils.io import load_pil_images
import random


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


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, vlprocessor, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        data_objs = []
        try:
            for data in data_path:
                # NOTE: load_dataset can process both json or jsonl files
                if data.endswith(".json") or data.endswith(".jsonl"):
                    data_objs.append(load_dataset("json", data_files=data, cache_dir=data_args.dataset_cache_dir)["train"])
                else:
                    raise Exception(f"Unsupported file format (<{data}>)!")
            list_data_dict = concatenate_datasets(data_objs)
        except:
            traceback.print_exc()
            # NOTE: compatible with the old version
            list_data_dict = []
            for data in data_path:
                if data.endswith(".json"):
                    data = json.load(open(data, "r"))
                    for i in data:
                        i['id'] = len(list_data_dict)
                        list_data_dict.append(i)
                elif data.endswith(".jsonl"):
                    with open(data, "r", encoding="utf-8") as fp:
                        for line in fp:
                            line = line.strip()
                            obj = json.loads(line)
                            obj["id"] = len(list_data_dict)
                            list_data_dict.append(obj)
                else:
                    raise Exception(f"Unsupported file format (<{data}>)!!!")

        print("Formatting inputs...Skip in lazy mode")
        self.vlprocessor = vlprocessor
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def _convert_normal(self, data_dict):
        data_folder = self.data_args.data_folder
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
                        if len(conv["value"]) > 1 and random.random() < self.data_args.short_cap:
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

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_dict = self.list_data_dict[i]
        data_dict = {
                "image_gen": ["/storage/jp/Janus/generated_samples/img_0.jpg"],
                "conversations": [
                {
                    "from": "human",
                    "value": ["A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair"]
                },
                {
                    "from": "gpt",
                    "value": ""
                }
        ]
        }

        # data_dict = {
        #         "image": ["/storage/jp/Janus/generated_samples/img_0.jpg"],
        #         "conversations": [
        #         {
        #             "from": "human",
        #             "value": "<image>\nWhat are the colors of the bus in the image?"
        #         },
        #         {
        #             "from": "gpt",
        #             "value": "The bus in the image is white and red."
        #         }
        # ]
        # }

        # data_dict = {
        #         "conversations": [
        #         {
        #             "from": "human",
        #             "value": "What are the colors of the bus in the image?"
        #         },
        #         {
        #             "from": "gpt",
        #             "value": "The bus in the image is white and red."
        #         }
        # ]
        # }

        try:
            modal, messages = self._convert_normal(data_dict)

            # load images and prepare for inputs
            pil_images = load_pil_images(messages)
            data_dict = self.vlprocessor(
                    conversations=messages, images=pil_images, force_batchify=True, is_training=True, modal=modal)
            data_dict['modals'] = [modal]

        except Exception as e:
            traceback.print_exc()
            backup_idx = random.randint(0, len(self.list_data_dict) - 1)
            print(f"Encounted error when process {i}-th example: {data_dict}, use {backup_idx}-th example instead!!!")
            return self.__getitem__(backup_idx)

        return data_dict


@dataclass
class DataCollatorWithFlatteningForSupervisedDataset(object):
    """Collate examples for batch flattened supervised fine-tuning."""

    vlprocessor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict], separator_id=-100) -> Dict[str, torch.Tensor]:
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        # new_input_ids = []
        # new_labels = []
        # position_ids = []
        # for idx in range(0, len(input_ids)):
        #     new_input_ids.append(input_ids[idx][:self.vlprocessor.tokenizer.model_max_length])
        #     temp_label = labels[idx][:self.vlprocessor.tokenizer.model_max_length]
        #     temp_label[0] = separator_id
        #     new_labels.append(temp_label)
        #     position_ids.append(torch.tensor(list(range(len(input_ids[idx][:self.vlprocessor.tokenizer.model_max_length])))))

        # new_input_ids = torch.cat(new_input_ids)
        # new_labels = torch.cat(new_labels)
        # position_ids = torch.cat(position_ids)


        # batch = dict(
        #     input_ids=new_input_ids,
        #     labels=new_labels,
        #     position_ids=position_ids,
        # )
        batch = dict()
                     
        # work for 'images' argument in `prepare_inputs_labels_for_multimodal`
        # batch["sft_format"] = torch.cat([x["sft_format"] for x in instances])
        batch["input_ids"] = torch.cat([x["input_ids"] for x in instances])
        batch["labels"] = torch.cat([x["labels"] for x in instances])
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances])
        batch["attention_mask"] = torch.cat([x["attention_mask"] for x in instances])
        batch["images_seq_mask"] = torch.cat([x["images_seq_mask"] for x in instances])
        batch["images_emb_mask"] = torch.cat([x["images_emb_mask"] for x in instances])
        batch["modals"] = sum([x["modals"] for x in instances], [])

        return batch
    


import torch
from torch.utils.data import DataLoader



if __name__ == "__main__":

    import sys; sys.path.append('/storage/jp/Janus')
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from arg_util import ModelArguments, DataArguments, TrainingArguments

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()



    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, cache_dir = './cache_dir')
    vlprocessor  = vl_chat_processor
    tokenizer = vl_chat_processor.tokenizer


    data_path = '/storage/dataset/filter_aes/final_coyo/coyo_part0_aes8639812.json'
    train_dataset = LazySupervisedDataset(
        vlprocessor=vl_chat_processor,
        data_path=data_path,
        data_args=data_args
    )

    # 创建 DataLoader
    batch_size = 8  # 你可以根据需要调整 batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 遍历 DataLoader 中的每一个 batch
    for batch_idx, batch in enumerate(train_dataloader):
        # batch 是一个包含多个 item 的 batch
        # 你可以在这里对每个 batch 进行处理
        print(f"Batch {batch_idx}:")
        print(batch)  # 打印 batch 内容

        # 如果你想处理每一个 item，可以在 batch 中进一步遍历
        for item_idx, item in enumerate(batch):
            print(f"Item {item_idx} in batch {batch_idx}:")
            print(item)  # 打印 item 内容
    
    # data_collator = DataCollatorWithFlatteningForSupervisedDataset(vlprocessor=vlprocessor)
    
    """
    
    export http_proxy=127.0.0.1:7896
    export https_proxy=127.0.0.1:7896


    conda activate janus_pro 
    python /storage/jp/Janus/trainer/dataset.py 
    
    
    """