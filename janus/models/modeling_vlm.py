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


class MultiModalityConfig(PretrainedConfig):
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
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

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
        self.scale_list = config.scale_list.split(',')
        self.scale_list = [ int(scale_) for scale_ in self.scale_list ]
        print(f'self.scale_list is:{self.scale_list}!')

        self.resize_transform_list = [
            transforms.Compose(
                                [
                                    transforms.Resize(
                                        ( 
                                        int(384 // (24/self.scale_list[i])), 
                                        int(384 // (24/self.scale_list[i]))
                                        )
                                    )
                                ]
                            )
            for i in range(len(self.scale_list))
        ]
        
        language_config._attn_implementation = config._attn_implementation_new
        print(f'_attn_implementation is:{language_config._attn_implementation}!')

        self.ar_with_non_ar = config.ar_with_non_ar
        self.is_causal = config.is_causal
        self.only_compute_ar_loss = config.only_compute_ar_loss
        self.visual_token_replace_max_ratio = config.visual_token_replace_max_ratio
        print(f'{self.visual_token_replace_max_ratio=}!')

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

    def random_replace(self, image_ids, max_ratio):
        """
        对 image_ids 的每条序列按 ratio 比例随机替换 ids。

        参数:
            image_ids: 输入的 ids 张量，shape 为 (b, seq)。
            ratio: 替换比例，范围 [0, 1]。

        返回:
            替换后的 ids 张量，shape 为 (b, seq)。
        """
        b, seq = image_ids.shape

        # 生成随机比例（若未提供 ratio）
        ratio = torch.rand(1).item() * max_ratio  # [0, 0.3)

        # 1. 确定需要替换的位置
        mask = torch.rand(b, seq) < ratio  # 生成随机掩码，shape 为 (b, seq)
        replace_indices = mask.nonzero(as_tuple=True)  # 获取需要替换的位置

        # 2. 生成新的 ids
        # 使用 torch.randint 随机生成新的 ids，范围是 [0, self.image_vocab_size - 1]
        new_ids = torch.randint(0, self.image_vocab_size, (b, seq), device=image_ids.device)

        # 3. 执行替换
        replaced_ids = image_ids.clone()  # 复制原始 ids
        replaced_ids[replace_indices] = new_ids[replace_indices]  # 替换选定的位置

        return replaced_ids
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

        # import ipdb; ipdb.set_trace()
        if 'image_gen' in modals:
            image_gen = True
            b, n = pixel_values.shape[0:2]
            ori_images = rearrange(pixel_values, "b n c h w -> (b n) c h w")  # 8, 3, 384, 384


            # 获取第一个尺度图片的vq index
            images = self.resize_transform_list[0](ori_images) # torch.Size([8, 3, 96, 96])
            z_q, (vq_loss, commit_loss, entropy_loss), (perplexity, min_encodings, min_encoding_indices) = self.gen_vision_model.encode(images)
            images_ids = min_encoding_indices.view(b * n, -1)

            # 获取第一个尺度文本、图片的tokens数目
            image_token_nums_first_stage = images_ids.size(1)
            text_token_nums = input_ids.size(1)
            ar_token_nums =  text_token_nums + image_token_nums_first_stage  

            # 获取图像和文本的embedding用于自回归
            img_embeds = self.prepare_gen_img_embeds(images_ids) # torch.Size([32, 36, 2048])
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            # LLM的输入embedding和label ids
            img_embeds_list = [img_embeds]
            labels_images_ids_list = [images_ids]

            labels[:, -1] = labels[:, -2]

            # 设置为None之后会自动取token loss平均, 需要注意pad token不能计算在内
            kwargs["num_items_in_batch"] = None # kwargs["num_items_in_batch"] * image_token_nums
  
            for scale_idx in range(2, len(self.scale_list)+1): #

                # 随机替换token
                images_ids_replaced = self.random_replace(images_ids, self.visual_token_replace_max_ratio)
                # 获取图像tokens被随机替换后embedding用于下一个尺度的输入
                img_embeds_replaced = self.prepare_gen_img_embeds(images_ids_replaced) # torch.Size([32, 36, 2048])


                # 训练中用前一个尺度的gt-embedding插值得到当前尺度的输入
                img_embeds_prev_embeds_replaced = img_embeds_replaced  # （b, n, c）
                (b, _, c) = img_embeds_prev_embeds_replaced.shape  
                img_embeds_prev_embeds_replaced = img_embeds_prev_embeds_replaced.view(b, self.scale_list[scale_idx-2], self.scale_list[scale_idx-2], c).permute(0, 3, 1, 2) # (bs, 2048, 6, 6)    
                img_embeds_curr_stage = F.interpolate(img_embeds_prev_embeds_replaced, size=(self.scale_list[scale_idx-1], self.scale_list[scale_idx-1]), mode='bilinear', align_corners=False) # (bs, 2048, 12, 12)
                img_embeds_curr_stage = img_embeds_curr_stage.permute(0, 2, 3, 1).view(b, self.scale_list[scale_idx-1]**2, c) # (bs, 144,  2048)
                # 插值后的拼接在输入后面
                img_embeds_list.append(img_embeds_curr_stage) # (b, q**2,  2048)


                # 当前阶段的gt真值
                images_curr_stage = self.resize_transform_list[scale_idx-1](ori_images)
                z_q, (vq_loss, commit_loss, entropy_loss), (perplexity, min_encodings, min_encoding_indices) = self.gen_vision_model.encode(images_curr_stage)
                images_ids_curr_stage = min_encoding_indices.view(b * n, -1)
                labels_images_ids_list.append(images_ids_curr_stage) # （b, q**2)

                # 下一个阶段的images_ids
                images_ids = images_ids_curr_stage
                
            # 所有的输入embeds拼一起
            img_embeds = torch.cat(img_embeds_list, dim=1)
            inputs_embeds = torch.cat([inputs_embeds, img_embeds], dim=1)
            
            # 所有的labels ids拼一起
            labels_images_ids = torch.cat(labels_images_ids_list, dim=1)
            labels = torch.cat([labels, labels_images_ids], dim=1)

            attention_mask = self.build_attention_mask(input_ids, labels_images_ids_list[0], labels_images_ids_list[1:], pad_token=100002).to(labels_images_ids.device) 
            attention_mask = (attention_mask.unsqueeze(1) - 1.)* torch.finfo(inputs_embeds.dtype).max
            attention_mask = attention_mask.to(inputs_embeds.dtype)

            if self.is_causal:
                attention_mask = None
            
            # attention_mask = None
            # import ipdb; ipdb.set_trace()
            # print("attention_mask",attention_mask)

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
                ar_token_nums=ar_token_nums,
                text_token_nums=text_token_nums,
                ar_with_non_ar = self.ar_with_non_ar,
                only_compute_ar_loss = self.only_compute_ar_loss,
                is_causal = self.is_causal,  
                **kwargs,
                )

            # ==============  第一个尺度图片重建来可视化 =================
            logits = output['logits']
            pred_scale1_ids = torch.argmax(logits[:, text_token_nums-1:ar_token_nums-1], dim=-1)  
            dec_scale1_recon = self.gen_vision_model.decode_code(
                pred_scale1_ids[0].to(dtype=torch.int),
                shape=[1, 8, self.scale_list[0], self.scale_list[0]]
            )
            dec_scale1_recon = dec_scale1_recon.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            
            dec_scale1_recon = np.clip((dec_scale1_recon + 1) / 2 * 255, 0, 255).astype(np.uint8)
            save_dir = '/storage/zhubin/Janus-zb/reconstructed_samples'; os.makedirs(save_dir, exist_ok=True)
            save_path_scale1_recon = os.path.join(save_dir, f"tmp_scale1_recon.jpg")
            PIL.Image.fromarray(dec_scale1_recon[0]).save(save_path_scale1_recon)

 
            # 第二个阶段重建
            if len(self.scale_list) > 1:
                pred_scale2_ids = torch.argmax(logits[ : , ar_token_nums-1 : ar_token_nums-1+self.scale_list[1]**2], dim=-1)  
                dec_scale2_recon = self.gen_vision_model.decode_code(
                    pred_scale2_ids[0].to(dtype=torch.int),
                    shape=[1, 8, self.scale_list[1], self.scale_list[1]]
                )
                dec_scale2_recon = dec_scale2_recon.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                dec_scale2_recon = np.clip((dec_scale2_recon + 1) / 2 * 255, 0, 255).astype(np.uint8)
                save_path_scale2_recon = os.path.join(save_dir, f"tmp_scale2_recon.jpg")
                PIL.Image.fromarray(dec_scale2_recon[0]).save(save_path_scale2_recon)
            
            # 第二个阶段重建---测试
            """pred_scale2_test_ids = torch.argmax(logits[ : , ar_token_nums : ar_token_nums+self.scale_list[1]**2], dim=-1)  

            dec_scale2_test_recon = self.gen_vision_model.decode_code(
                pred_scale2_test_ids[0].to(dtype=torch.int),
                shape=[1, 8, 24, 24]
            )
            dec_scale2_test_recon = dec_scale2_test_recon.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
           
            dec_scale2_test_recon = np.clip((dec_scale2_test_recon + 1) / 2 * 255, 0, 255).astype(np.uint8)
            save_path_scale2_test_recon = os.path.join('/storage/jp/Janus/generated_samples_0225', f"tmp_scale2_test_recon.jpg")
            PIL.Image.fromarray(dec_scale2_test_recon[0]).save(save_path_scale2_test_recon)"""
            

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
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
