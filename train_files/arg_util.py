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
    gen_vision_cls: Optional[str] = field(default="Coscom_DV4x8x8")
    gen_vision_image_token_size : int = field(
        default=64000,
    )
    gen_vision_n_embed: int = field(
        default= 6 ,
    )

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

    image_under_data_files: Optional[str] = field(default='/storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/mini_uni_DenseFusion-4V-100k.json')
    image_under_rootdir: Optional[str] = field(default='/storage/yqs/dataset/BAAI/DenseFusion-1M/images')
    image_gen_data_files: Optional[str] = field(default='/storage/dataset/filter_aes/cap_merge_final_640/recap2/mini_janus_part0_cap6595998.json')
    image_gen_rootdir: Optional[str] = field(default='/storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded')
    text_chat_data_files: Optional[str] = field(default='/storage/yqs/dataset/BAAI/Infinity-Instruct/7M_domains/subjective/mini_output.json')
    sample_rate: Optional[str] = field(default=None)
    samples_per_epoch: int = field(
        default=10000,
        metadata={"help": "samples_per_epoch."}
    )
    batchsize_list: Optional[str] = field(default='1,1,1')
    dataset: Optional[str] = field(default="image_under||image_gen||text_chat")


@dataclass
class TrainingArguments(transformers.TrainingArguments):

    report_to : List[str] = field(default=None)
    # ar_with_non_ar: bool = field(default=True)
    # mask_ar_image: bool = field(default=False)
    # is_causal: bool = field(default=False)
    # only_compute_ar_loss: bool = field(default=False)
    # only_compute_non_ar_loss: bool = field(default=False)
    _attn_implementation_new: str = field(default="flash_attention_2")
    # scale_list : str = field(default="12,24")
    # visual_token_replace_max_ratio: Optional[float] = None
    # shut auto processing (_remove_unused_columns) of transformers Trainer
    remove_unused_columns: bool = field(default=False)

    first_init: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    # Training learning rate Arguments
    gen_vision_encoder_lr: Optional[float] = None
    vision_encoder_lr: Optional[float] = None
    aligner_lr: Optional[float] = None
    llm_lr: Optional[float] = None
    gen_aligner_lr: Optional[float] = None
    gen_head_lr: Optional[float] = None
    gen_embed_lr: Optional[float] = None
    lm_pos_lr: Optional[float] = None
    multi_scale_lr: Optional[float] = None

    # Training Data Arguments
    use_tokenizer_pad:  bool = field(
        default=False,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    tokenizer_model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum tokenizer sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

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
    dataloader_prefetch_factor: int =  16
    