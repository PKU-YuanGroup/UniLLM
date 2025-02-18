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
from packaging import version
from torch.utils.data import Dataset
from arg_util import ModelArguments, DataArguments, TrainingArguments
from dataset import LazySupervisedDataset, DataCollatorWithFlatteningForSupervisedDataset
from trainer import JanusTrainer, safe_save_model_for_hf_trainer
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

sys.path.append('./')
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.modeling_vlm import MultiModalityConfig

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
    train_dataset = LazySupervisedDataset(
        vlprocessor=vlprocessor,
        data_path=data_args.data_path,
        data_args=data_args
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

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # BUG: High version transformers report error:
            # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    config = MultiModalityConfig.from_pretrained(model_args.model_path)
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_args.model_path, trust_remote_code=True, config=config,
    )

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model.language_model = prepare_model_for_kbit_training(model.language_model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model.language_model, "enable_input_require_grads"):
            model.language_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # decoupled learning rate
    model.config.llm_lr = training_args.llm_lr
    model.config.vision_encoder_lr = training_args.vision_encoder_lr
    model.config.aligner_lr = training_args.aligner_lr
    model.config.gen_aligner_lr = training_args.gen_aligner_lr
    model.config.gen_head_lr = training_args.gen_head_lr
    model.config.gen_embed_lr = training_args.gen_embed_lr

    if model.config.llm_lr is None:
        for p in model.language_model.parameters():
            p.requires_grad = False

    if model.config.vision_encoder_lr is None:
        for p in model.vision_model.parameters():
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


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if local_rank == 0:
        print("Current model:", model)
        print("Model config:", model.config)

    if data_args.use_batch_flattening:
        rank0_print('You are using flattening operation to flatten the entire mini batch into a single sequence')
        assert model.language_model.config._attn_implementation == 'flash_attention_2'
        assert version.parse(transformers.__version__) >= version.parse("4.44.0")
        data_module = make_flattening_supervised_data_module(vlprocessor=vl_chat_processor, data_args=data_args)
    else:
        data_module = make_supervised_data_module(vlprocessor=vl_chat_processor, data_args=data_args)

    # select a Trainer
    trainer = JanusTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()