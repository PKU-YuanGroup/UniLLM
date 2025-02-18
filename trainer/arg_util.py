import os
import re
import sys
import transformers
from packaging import version
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


def int_with_none(value):
    if value == 'None':
        return None
    return int(value)


@dataclass
class ModelArguments:
    # LLM Arguments
    model_type: Optional[str] = field(default="videollama3", metadata={"help": "Model type selected in the list: "})
    model_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    # mm_projector_type: Optional[str] = field(default='linear')
    # pretrain_mm_projector: Optional[str] = field(default=None)
    # Vision tower Arguments
    # vision_encoder: Optional[str] = field(default=None)
    # mm_vision_select_layer: Optional[int] = field(default=-1)
    # mm_vision_select_feature: Optional[str] = field(default="patch")
    # mm_attn_implementation: Optional[str] = field(default="flash_attention_2")
    # Token downsampling Arguments
    # use_token_compression: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    # Path Arguments
    data_path: List[str] = field(default=None, metadata={"help": "Path to the training data."})
    # image_folder: Optional[str] = field(default=None)
    # video_folder: Optional[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    # Loading Arguments
    use_batch_flattening: bool = field(default=True, metadata={"help": "Whether to flatten the in-batch sequences of variable lengths."})
    dataset_cache_dir: Optional[str] = field(default=None)
    short_cap: Optional[float] = field(default=0.2)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # shut auto processing (_remove_unused_columns) of transformers Trainer
    remove_unused_columns: bool = field(default=False)

    optim: str = field(default="adamw_torch")
    # Training learning rate Arguments
    vision_encoder_lr: Optional[float] = None
    aligner_lr: Optional[float] = None
    llm_lr: Optional[float] = None
    gen_aligner_lr: Optional[float] = None
    gen_head_lr: Optional[float] = None
    gen_embed_lr: Optional[float] = None

    # Training Data Arguments
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"