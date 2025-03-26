# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    # LlamaConfig,
    # LlamaForCausalLM,
    PreTrainedModel,
)


 
from janus.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
import numpy as np, os; import PIL.Image
from typing import Callable, List, Optional, Tuple, Union
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from torchvision import transforms
from transformers.cache_utils import Cache
from torch.nn import functional as F
# /storage/miniconda3/envs/janus_pro/lib/python3.10/site-packages/transformers/modeling_outputs.py

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector

# ADD 
import mediapy as media
from .cosmos_tokenizer.utils import numpy2tensor, tensor2numpy

class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower
    
    # ADD
    elif "Cosmos" in cls_name:
        from janus.models.cosmos_tokenizer.video_lib import CausalVideoTokenizer
        # import ipdb;ipdb.set_trace()
        # input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
        
        # encoder = CausalVideoTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
        # decoder = CausalVideoTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')
        cls =  CausalVideoTokenizer 

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]   
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig_SepVL(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig_SepVL
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM_SepVL(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig_SepVL):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        if gen_vision_config.cls == 'Cosmos_DV4x8x8':
            model_name = "Cosmos-0.1-Tokenizer-DV4x8x8"
            import sys; cur_dir = os.path.dirname(os.path.abspath(__file__)); sys.path.append(os.path.join(cur_dir, '../../'))
            self.gen_vision_model = gen_vision_cls(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
            self.gen_vision_model_decoder = gen_vision_cls(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')
        elif gen_vision_config.cls == 'Cosmos_DV8x16x16':
            model_name = "Cosmos-0.1-Tokenizer-DV8x16x16"
            import sys; cur_dir = os.path.dirname(os.path.abspath(__file__)); sys.path.append(os.path.join(cur_dir, '../../'))
            self.gen_vision_model = gen_vision_cls(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
            self.gen_vision_model_decoder = gen_vision_cls(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')
        else:
            self.gen_vision_model = gen_vision_cls()
        # encoder = CausalVideoTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
        # decoder = CausalVideoTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

        # ADD 
        self.image_vocab_size = gen_vision_config.params.image_token_size # self.gen_embed -- shape[0]
    
        self.resize_transform =   transforms.Compose(
                                [
                                    transforms.Resize(    (   384, 384  )   )
                                ]
                            )
        if hasattr(config, "_attn_implementation_new"):
            language_config._attn_implementation = config._attn_implementation_new
        else:
            language_config._attn_implementation =  'flash_attention_2'
        print(f'_attn_implementation is:{language_config._attn_implementation}!')
 

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        # 调用语言模型的gradient_checkpointing_enable方法，并传递梯度检查点参数
        # gradient_checkpointing_kwargs: 包含梯度检查点配置的字典或参数
        return self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def build_attention_mask(self, input_ids, img1_ids, img_ids_list, pad_token=100002):
        # 获取序列长度
        bs = input_ids.shape[0] # batch size
        text_len = input_ids.shape[1]  # 文本长度
        img1_len = img1_ids.shape[1]  # img1 长度
        img_lens = [img_ids.shape[1] for img_ids in img_ids_list]  # 其他 img_ids 的长度
        seq_len = text_len + img1_len + sum(img_lens)  # 总长度

        # 初始化 attention_mask 为全 0
        attention_mask = torch.zeros(bs, seq_len, seq_len).to(input_ids.device)

        # 构建文本和 img1_ids 的因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角矩阵
        attention_mask[:, :text_len + img1_len, :text_len + img1_len] = causal_mask[:text_len + img1_len, :text_len + img1_len]

        # 构建每个 img_ids 的内部可见掩码
        start_idx = text_len + img1_len  # img2_ids 的起始位置
        for i, img_len in enumerate(img_lens):
            end_idx = start_idx + img_len
            # img_i_ids 可以看到文本和所有之前的 img_ids
            attention_mask[:, start_idx:end_idx, :end_idx] = 1
            # img_i_ids 内部相互可见
            attention_mask[:, start_idx:end_idx, start_idx:end_idx] = 1
            start_idx = end_idx  # 更新起始位置

        # --- 新增部分：处理 pad_token 的掩码 ---
        # 1. 识别 input_ids 中的 pad_token 位置（假设 pad_token 在 input_ids 的前部）
        pad_mask = (input_ids ==  pad_token)  # shape: (bs, text_len)

        # 2. 将 pad_mask 扩展到完整序列长度（填充部分默认非 pad_token）
        full_pad_mask = torch.zeros(bs, seq_len, dtype=torch.bool, device=input_ids.device)
        full_pad_mask[:, :text_len] = pad_mask  # 仅文本部分可能有 pad_token

        # 3. 对 attention_mask 应用 pad_token 的掩码：
        #    - pad_token 所在行（不能关注任何位置）
        #    - pad_token 所在列（其他位置不能关注它）
        attention_mask.masked_fill_(
            full_pad_mask.unsqueeze(1) | full_pad_mask.unsqueeze(2), 
            0
        )

        return attention_mask
    
    
    def build_attention_mask_v2(self, input_ids, img1_ids, img_ids_list, pad_token=100002):
        # 获取序列长度
        bs = input_ids.shape[0] # batch size
        text_len = input_ids.shape[1]  # 文本长度
        img1_len = img1_ids.shape[1]  # img1 长度
        img_lens = [img_ids.shape[1] for img_ids in img_ids_list]  # 其他 img_ids 的长度
        seq_len = text_len + img1_len + sum(img_lens)  # 总长度

        # 初始化 attention_mask 为全 0
        attention_mask = torch.zeros(bs, seq_len, seq_len).to(input_ids.device)

        # 构建文本和 img1_ids 的因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角矩阵
        attention_mask[:, :text_len + img1_len, :text_len + img1_len] = causal_mask[:text_len + img1_len, :text_len + img1_len]

        # 构建每个 img_ids 的内部可见掩码
        start_idx = text_len + img1_len  # img2_ids 的起始位置
        for i, img_len in enumerate(img_lens):
            end_idx = start_idx + img_len
            # img_i_ids 可以看到文本和所有之前的 img_ids
            attention_mask[:, start_idx:end_idx, :end_idx] = 1
            start_idx = end_idx  # 更新起始位置

        # 第一个scale是自回归的，后面非自回归的scale不允许看到真值
        ar_image_start_idx = text_len
        ar_image_end_idx = text_len + img1_len
        attention_mask[:, ar_image_end_idx:, ar_image_start_idx:ar_image_end_idx] = 0
        # --- 新增部分：处理 pad_token 的掩码 ---
        # 1. 识别 input_ids 中的 pad_token 位置（假设 pad_token 在 input_ids 的前部）
        pad_mask = (input_ids ==  pad_token)  # shape: (bs, text_len)

        # 2. 将 pad_mask 扩展到完整序列长度（填充部分默认非 pad_token）
        full_pad_mask = torch.zeros(bs, seq_len, dtype=torch.bool, device=input_ids.device)
        full_pad_mask[:, :text_len] = pad_mask  # 仅文本部分可能有 pad_token

        # 3. 对 attention_mask 应用 pad_token 的掩码：
        #    - pad_token 所在行（不能关注任何位置）
        #    - pad_token 所在列（其他位置不能关注它）
        attention_mask.masked_fill_(
            full_pad_mask.unsqueeze(1) | full_pad_mask.unsqueeze(2), 
            0
        )

        return attention_mask
    
    def initialize_moe_modules(self, config):
        hidden_size= self.config.language_config.hidden_size
        from deepspeed.moe.layer import MoE

        moe_layers = [1,5,9,13,17,21,25,29,33,37,41]

        for i, layer in enumerate(self.language_model.model.layers):
            if i in moe_layers:
                pretrained_state_dict_text = layer.mlp_text.state_dict()
                layer.mlp_text = MoE(
                    hidden_size,
                    expert=layer.mlp_text,
                    num_experts=config.moe_num_experts_text,
                    ep_size=config.moe_ep_size_text,
                    k=config.moe_top_k_experts_text,
                    capacity_factor=1.5,
                    eval_capacity_factor=2.0,
                    min_capacity=0,
                    use_residual=False,
                    enable_expert_tensor_parallelism=True,
                )
                for e in layer.mlp_text.deepspeed_moe.experts.deepspeed_experts:  # check weight
                    loaded_state_dict = e.state_dict()
                    assert all([torch.allclose(pretrained_state_dict_text[k], v) for k, v in loaded_state_dict.items()])
                    assert all([torch.allclose(loaded_state_dict[k], v) for k, v in pretrained_state_dict_text.items()])
        for i, layer in enumerate(self.language_model.model.layers):
            if i in moe_layers:
                pretrained_state_dict_vision = layer.mlp_vision.state_dict()
                layer.mlp_vision = MoE(
                    hidden_size,
                    expert=layer.mlp_vision,
                    num_experts=config.moe_num_experts_vision,
                    ep_size=config.moe_ep_size_vision,
                    k=config.moe_top_k_experts_vision,
                    capacity_factor=1.5,
                    eval_capacity_factor=2.0,
                    min_capacity=0,
                    use_residual=False,
                    enable_expert_tensor_parallelism=True,
                )
                for e in layer.mlp_vision.deepspeed_moe.experts.deepspeed_experts:  # check weight
                    loaded_state_dict = e.state_dict()
                    assert all([torch.allclose(pretrained_state_dict_vision[k], v) for k, v in loaded_state_dict.items()])
                    assert all([torch.allclose(loaded_state_dict[k], v) for k, v in pretrained_state_dict_vision.items()])

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,

        # ADD
        pixel_values: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_emb_mask: Optional[torch.FloatTensor] = None,
        modals: Optional[List[str]] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python

        ```"""


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if "image" in modals and inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                pixel_values=pixel_values,
                images_seq_mask=images_seq_mask,
                images_emb_mask=images_emb_mask,
            )
            input_ids = None

            return self.language_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
                )

        elif "text" in modals:
            # print("models!!",modals)
            return self.language_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
                )
        
        # import ipdb; ipdb.set_trace() 
        elif 'image_gen' in modals:
            image_gen = True
            b, n = pixel_values.shape[0:2]
            ori_images = rearrange(pixel_values, "b n c h w -> (b n) c h w")  # 8, 3, 384, 384


            # 获取第一个尺度图片的vq index
            images = self.resize_transform(ori_images) # torch.Size([8, 3, 96, 96])

            # import ipdb; ipdb.set_trace()
            # coscom
            if 'Cosmos' in self.config.gen_vision_config.cls:
                encoder = self.gen_vision_model
                if images.dim() == 4:
                    images = images.unsqueeze(2)
                (indices, codes) = encoder.encode(images) 
                b, t, h, w = indices.shape
                images_ids = indices.view(b, -1)
            else:
                z_q, (vq_loss, commit_loss, entropy_loss), (perplexity, min_encodings, min_encoding_indices) = self.gen_vision_model.encode(images)
                images_ids = min_encoding_indices.view(b * n, -1)

            # 获取第一个尺度文本、图片的tokens数目  
            image_token_nums = images_ids.size(1)
            text_token_nums = input_ids.size(1)

            # 获取图像和文本的embedding用于自回归
            # import ipdb; ipdb.set_trace()
            img_embeds = self.prepare_gen_img_embeds(images_ids) # torch.Size([32, 36, 2048])
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            # LLM的输入embedding和label ids
            labels[:, -1] = labels[:, -2]
            # 设置为None之后会自动取token loss平均, 需要注意pad token不能计算在内
            kwargs["num_items_in_batch"] =  kwargs["num_items_in_batch"] * image_token_nums

            # import ipdb; ipdb.set_trace()
            # 所有的输入embeds拼一起
            inputs_embeds = torch.cat([inputs_embeds, img_embeds], dim=1)
            # 所有的labels ids拼一起
            labels = torch.cat([labels, images_ids], dim=1)

            # attention_mask = self.build_attention_mask(input_ids, images_ids, [], pad_token=100002).to(images_ids.device) 
            # attention_mask = (attention_mask.unsqueeze(1) - 1.)* torch.finfo(inputs_embeds.dtype).max
            # attention_mask = attention_mask.to(inputs_embeds.dtype)

            # if self.is_causal:
            attention_mask = None
    
            # /storage/zhubin/Janus-zb/janus/models/llama/modeling_llama.py
            output =  self.language_model.forward(
                input_ids=None,
                attention_mask=attention_mask, # torch.Size([32, 28])
                # attention_mask=None, # torch.Size([32, 28]) #attention_mask
                position_ids=position_ids, # None
                past_key_values=past_key_values, # None
                inputs_embeds=inputs_embeds, # torch.Size([32, 64, 2048])
                labels=labels, # torch.Size([32, 64])
                use_cache=False, # use_cache
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,  
                cache_position=cache_position,
                logits_to_keep = logits_to_keep,
                # ADD
                image_gen=image_gen,
                vocab_size=self.image_vocab_size,
                gen_head=self.gen_head,
                image_token_nums=image_token_nums,
                **kwargs,
                )

            # ==============  第一个尺度图片重建来可视化 =================
            logits = output['logits']
            pred_scale1_ids = torch.argmax(logits[:, text_token_nums-1:-1], dim=-1)  
            
            # import ipdb; ipdb.set_trace()
            if 'Cosmos' in self.config.gen_vision_config.cls:
                pred_scale1_ids = pred_scale1_ids.reshape(indices.shape)
                reconstructed_tensor = self.gen_vision_model_decoder.decode(pred_scale1_ids)
                reconstructed_tensor = reconstructed_tensor.squeeze(2) 

                # reconstructed_tensor = reconstructed_tensor.permute(1,2,0) 
                recon_image = tensor2numpy(reconstructed_tensor)[0]
                save_dir = '/storage/zhubin/Janus-MoE/reconstructed_samples'; os.makedirs(save_dir, exist_ok=True)
                save_path_scale1_recon = os.path.join(save_dir, f"scale1_recon.jpg")
                media.write_image(save_path_scale1_recon, recon_image)
            else:
                dec_scale1_recon = self.gen_vision_model.decode_code(
                pred_scale1_ids[0].to(dtype=torch.int),
                    shape=[1, 8, 24, 24]
                )
                dec_scale1_recon = dec_scale1_recon.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                
                dec_scale1_recon = np.clip((dec_scale1_recon + 1) / 2 * 255, 0, 255).astype(np.uint8)
                save_dir = '/storage/zhubin/Janus-MoE/reconstructed_samples'; os.makedirs(save_dir, exist_ok=True)
                save_path_scale1_recon = os.path.join(save_dir, f"scale1_recon.jpg")
                PIL.Image.fromarray(dec_scale1_recon[0]).save(save_path_scale1_recon)
            


            """


       
            
            """

            return output
 
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

       

AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig_SepVL)
AutoModelForCausalLM.register(MultiModalityConfig_SepVL, MultiModalityCausalLM_SepVL)
