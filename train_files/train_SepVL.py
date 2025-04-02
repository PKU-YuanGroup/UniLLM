# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math
import copy
import json
import os
import pathlib
import random
import re
import sys
import warnings
import traceback
from packaging import version
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
import transformers
from dataclasses import dataclass, field

from packaging import version
from torch.utils.data import Dataset
from arg_util import ModelArguments, DataArguments, TrainingArguments
# from dataset import LazySupervisedDataset, DataCollatorWithFlatteningForSupervisedDataset

sys.path.append('./')
from train_files.dataset_v2  import  UniDataset, DataCollatorWithFlatteningForSupervisedDataset

from trainer import JanusTrainer, safe_save_model_for_hf_trainer
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

from janus.models import  VLChatProcessor
# from janus.models.modeling_vlm import  MultiModalityCausalLM 
from janus.models.modeling_vlm_SepVL  import  MultiModalityCausalLM_SepVL
from janus.models.modeling_vlm_SepVL import MultiModalityConfig_SepVL


from attrdict import AttrDict
# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486
os.environ["TOKENIZERS_PARALLELISM"] = "true"
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)



def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_flattening_supervised_data_module(vlprocessor: transformers.ProcessorMixin, data_args) -> Dict:
    """Make batch flattened dataset and collator for supervised fine-tuning."""
 
    data_args.image_under_data_files = [ filepath.strip()  for filepath in  data_args.image_under_data_files.split(',') ]
    data_args.image_gen_data_files =   [ filepath.strip()  for filepath in  data_args.image_gen_data_files.split(',') ]
    data_args.text_chat_data_files =   [ filepath.strip()  for filepath in  data_args.text_chat_data_files.split(',') ]
    data_args.sample_rate = data_args.sample_rate.split(','); data_args.sample_rate = [ int(i) for i in data_args.sample_rate]
    data_args.batchsize_list = data_args.batchsize_list.split(','); data_args.batchsize_list = [ int(i) for i in data_args.batchsize_list]
    # data_args.dataset = data_args.dataset.split(',')

    print(f'{data_args=}!!!')

    train_dataset = UniDataset(
        vlprocessor=vlprocessor,
        image_under_data_files=data_args.image_under_data_files, # ['/storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/mini_uni_DenseFusion-4V-100k.json'], 
        image_under_rootdir=data_args.image_under_rootdir, #'/storage/yqs/dataset/BAAI/DenseFusion-1M/images',
        image_gen_data_files=data_args.image_gen_data_files, #['/storage/dataset/filter_aes/cap_merge_final_640/recap2/mini_janus_part0_cap6595998.json'],
        image_gen_rootdir=data_args.image_gen_rootdir, #'/storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded',
        text_chat_data_files=data_args.text_chat_data_files, #['/storage/yqs/dataset/BAAI/Infinity-Instruct/7M_domains/subjective/mini_output.json'],
        samples_per_epoch=data_args.samples_per_epoch, #100000,
        dataset=data_args.dataset, #"image_under||image_gen||text_chat",
        sample_rate=data_args.sample_rate, #[5, 4, 1],
        batchsize_list=data_args.batchsize_list #[1,1,1]
    )


    data_collator = DataCollatorWithFlatteningForSupervisedDataset(vlprocessor=vlprocessor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

                
def train():
    global local_rank
    set_seed(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    if local_rank == 0:
        print('------model args------')
        print(model_args)
        print('------data args------')
        print(data_args)
        print('------training args------')
        print(training_args)

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    model_args.torch_dtype = compute_dtype


    config = MultiModalityConfig_SepVL.from_pretrained(model_args.model_path, cache_dir='./cache_dir')
    setattr(config, '_attn_implementation_new', training_args._attn_implementation_new)

    # ADD  gen_vision_model
    setattr(config.gen_vision_config, 'cls', model_args.gen_vision_cls)
    setattr(config.gen_vision_config.params, 'image_token_size', model_args.gen_vision_image_token_size)
    setattr(config.gen_vision_config.params, 'n_embed', model_args.gen_vision_n_embed)
 
    # ADD gen_head
    setattr(config.gen_head_config.params, 'image_token_size', model_args.gen_vision_image_token_size)
 
    # ADD languages
    setattr(config.language_config, 'max_position_embeddings', model_args.gen_vision_image_token_size)
 
    # aligner
    setattr(config.gen_aligner_config.params, 'input_dim', model_args.gen_vision_n_embed)

    # import ipdb; ipdb.set_trace()
    # vision_diff
    
    vision_diff_params = {
        'patch_size': model_args.diff_patch_size,
        'target_channels': config.language_config.hidden_size * model_args.diff_patch_size**2 ,
        'z_channels': config.language_config.hidden_size,
        'diffloss_w': config.language_config.hidden_size,
        'diffloss_d': model_args.diffloss_d,
        'num_sampling_steps': model_args.diff_num_sampling_steps,
        'grad_checkpointing': model_args.diff_grad_checkpointing,
        'diffusion_batch_mul': model_args.diffusion_batch_mul,
    }
    vision_diff_params = AttrDict(vision_diff_params)
    # vision_diff_params = {
    #     'target_channels': 2048,
    #     'z_channels': 2048,
    #     'diffloss_w': 2048 ,
    #     'diffloss_d': 3 ,
    #     'num_sampling_steps': "100",
    #     'grad_checkpointing': True
    # }
    setattr(config.vision_diff_config, 'params', vision_diff_params)


    # import ipdb; ipdb.set_trace()
    # model: MultiModalityCausalLM_SepVL = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_path, trust_remote_code=True, config=config,  cache_dir='./cache_dir'
    # )

    # 只加载模型框架
    # model = MultiModalityCausalLM_SepVL.from_pretrained(
    #     model_args.model_path, trust_remote_code=True, config=config,  cache_dir='./cache_dir',
    #     state_dict=None  # 不加载权重
    # )

    # 只加载模型框架
    model = MultiModalityCausalLM_SepVL(
          config=config 
    )

    # import ipdb; ipdb.set_trace()
    # ============== 继承权重 ======================
    def inherit_weights(model, 
                        model_path=None,
                        model_path2=None):
        # 加载 Hugging Face 格式的预训练模型
        # import ipdb; ipdb.set_trace()
        pretrained_dict = torch.load(model_path, map_location="cpu")  # 加载到 CPU
        if model_path2 is not None:
            pretrained_dict2 = torch.load(model_path2, map_location="cpu")
            pretrained_dict.update(pretrained_dict2)
        model_dict = model.state_dict()  # 获取当前模型的权重
        # import ipdb; ipdb.set_trace()

        for key in model_dict:
            if '_text' in key :
                # 如果 key 以 '_text' 结尾，去掉 '_text' 并从 pretrained_dict 中获取对应的权重
                pretrained_key = key.replace('_text', '')
                if pretrained_key in pretrained_dict:
                    model_dict[key] = pretrained_dict[pretrained_key]
                # print(f'11111111111111', key)
            elif '_vision.' in  key and 'gen_vision' not in key:
                # 如果 key 以 '_vision' 结尾，去掉 '_vision' 并从 pretrained_dict 中获取对应的权重
                pretrained_key = key.replace('_vision', '')
                if pretrained_key in pretrained_dict:
                    model_dict[key] = pretrained_dict[pretrained_key]
                # print(f'22222222222222', key)
            elif 'gen_vision' in key or 'gen_head' in key or 'gen_embed' in key or 'gen_aligner' in key:
                # 换个vqvae, 这一部分不要了, 保持cosmos vqvae的原参数
                pass
            elif 'diffloss' in key:
                # diff loss没有预训练权重
                pass
            else:
                try:
                    model_dict[key] = pretrained_dict[key] # model_dict[key] == pretrained_dict[key]
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                    a=1
                    """
                    aa = list(pretrained_dict.keys())
                    with open('pretrained_dict.json', 'w', encoding='utf-8') as f:
                        json.dump(aa, f,  indent=2)

                    aa = list(pretrained_dict2.keys())
                    with open('pretrained_dict2.json', 'w', encoding='utf-8') as f:
                        json.dump(aa, f,  indent=2)
                    """
                # print(f'33333333333333')
        
        # 将更新后的权重加载到模型中

        msg = model.load_state_dict(model_dict, strict=True)
        print(msg, '!!!!!!!!!!!!!!!')
        return model


    if '7B' in model_args.model_path:
        model = inherit_weights(model, 
                                f'{model_args.model_path}/pytorch_model-00001-of-00002.bin', 
                                f'{model_args.model_path}/pytorch_model-00002-of-00002.bin')
    elif '1B' in model_args.model_path:
        model = inherit_weights(model, f'{model_args.model_path}/pytorch_model.bin')

    if training_args.moe:
        model.initialize_moe_modules(training_args)

    if training_args.gradient_checkpointing:
        if hasattr(model.language_model, "enable_input_require_grads"):
            model.language_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

 
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_args.model_path )
    vl_chat_processor.tokenizer_model_max_length = model_args.tokenizer_model_max_length
    vl_chat_processor.use_tokenizer_truncation =  model_args.use_tokenizer_truncation



    tokenizer = vl_chat_processor.tokenizer
    assert tokenizer.pad_token is not None
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.unk_token

    # decoupled learning rate
    model.config.llm_lr = training_args.llm_lr 
    model.config.vision_encoder_lr = training_args.vision_encoder_lr
    model.config.aligner_lr = training_args.aligner_lr
    model.config.gen_vision_encoder_lr = training_args.gen_vision_encoder_lr
    model.config.gen_aligner_lr = training_args.gen_aligner_lr
    model.config.gen_head_lr = training_args.gen_head_lr
    model.config.gen_embed_lr = training_args.gen_embed_lr
 
 
    if model.config.llm_lr is None:
        for p in model.language_model.parameters():
            p.requires_grad = False
 
    if model.config.vision_encoder_lr is None:
        for p in model.vision_model.parameters():
            p.requires_grad = False
  
    if model.config.gen_vision_encoder_lr is None:
        for p in model.gen_vision_model.parameters():
            p.requires_grad = False

        # ADD
        if hasattr(model, 'gen_vision_model_decoder'):
            for p in model.gen_vision_model_decoder.parameters():
                p.requires_grad = False


    if model.config.aligner_lr is None:
        for p in model.aligner.parameters():
            p.requires_grad = False

    if model.config.gen_aligner_lr is None:
        for p in model.gen_aligner.parameters():
            p.requires_grad = False

    if model.config.gen_head_lr is None:
        for p in model.gen_head.parameters():
            p.requires_grad = False
            
    if model.config.gen_embed_lr is None:
        for p in model.gen_embed.parameters():
            p.requires_grad = False

    if local_rank == 0:
        # 使用rank 0的进程写入一次
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(f'{training_args.output_dir}/trainv2_model_parameters.txt', 'w') as f:
            # 遍历所有参数及其是否需要训练
            for name, param in model.named_parameters():
                f.write(f"Parameter name: {name}\n")
                f.write(f"Requires grad: {param.requires_grad}\n")
                f.write("-" * 40 + "\n")
        print("Parameters have been logged to trainv2_model_parameters.txt")

        

    if local_rank == 0:
        print("Current model:", model)
        print("Model config:", model.config)

    if data_args.use_batch_flattening:
        rank0_print('You are using flattening operation to flatten the entire mini batch into a single sequence')
        # assert model.language_model.config._attn_implementation == 'flash_attention_2'
        assert version.parse(transformers.__version__) >= version.parse("4.44.0")
        data_module = make_flattening_supervised_data_module(vlprocessor=vl_chat_processor, data_args=data_args)
 
    # select a Trainer
    trainer = JanusTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()