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
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image

from torch.nn import functional as F

# specify the path to the model
model_path = "/storage/jp/Janus/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer


# model_path = "/storage/zhubin/Janus-zb/checkpoints/stage1_1scale_192_sdpa_ft/videollama3_qwen2.5_2b/stage_1/checkpoint-10000"
# model_path = "/storage/zhubin/Janus-zb/checkpoints/stage1_1scale_192_sdpa_ft_attnmask/videollama3_qwen2.5_2b/stage_1/checkpoint-1000"
model_path = "/storage/zhubin/Janus-zb/checkpoints/stage1_2scale_192_384_sdpa_ft_attnmask_random_replace/videollama3_qwen2.5_2b/stage_1/checkpoint-34000"
model_path = "/storage/zhubin/Janus-zb/checkpoints/stage1_2scale_192_384_sdpa_ft_attnmask_random_replace_0.7/videollama3_qwen2.5_2b/stage_1/checkpoint-2000"
save_image_dir = 'useless/generated_samples_stage1_2scale_192_384_sdpa_ft_attnmask_random_replace_0.7'
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


conversation = [
    {
        "role": "User",
        "content": "A man.",
    },
    {"role": "Assistant", "content": ""},
]

# import ipdb; ipdb.set_trace()


sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 144,
    img_size: int = 192,
    patch_size: int = 16,
    scale_list = None,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens) # tokens.shape torch.Size([2*p, 11, 2048])
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state # (p, 144)
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :]) # (2*p, 16384)
        logit_cond = logits[0::2, :] # (p, 16384)
        logit_uncond = logits[1::2, :] # (p, 16384)
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond) # (p, 16384)
        probs = torch.softmax(logits / temperature, dim=-1) # (p, 16384)

        next_token = torch.multinomial(probs, num_samples=1) # (p, 1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1) 

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1) # (p, 2)-> (p*2)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token) # (p*2, 2048)
        inputs_embeds = img_embeds.unsqueeze(dim=1) # (p*2, 1, 2048)

    # import ipdb; ipdb.set_trace()
    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec


    
    os.makedirs(save_image_dir, exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join(save_image_dir, "img_{}_192.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)


    # ====================  后续阶段  ===========================
    

    # inputs_embeds = torch.cat([inputs_embeds, img_embeds], dim=1)
    generated_img_embeds = mmgpt.prepare_gen_img_embeds(generated_tokens)

    # import ipdb; ipdb.set_trace()
    for scale_idx in range(2, len(scale_list)+1): #
        
        img_size = patch_size * scale_list[scale_idx-1] 
        # 训练中用前一个尺度的gt-embedding插值得到当前尺度的输入
        img_embeds_prev_gt = generated_img_embeds  # （b, n, c）
        (b, _, c) = img_embeds_prev_gt.shape  
        img_embeds_prev_gt = img_embeds_prev_gt.view(b, scale_list[scale_idx-2], scale_list[scale_idx-2], c).permute(0, 3, 1, 2) # (bs, 2048, 6, 6)    
        img_embeds_curr_stage = F.interpolate(img_embeds_prev_gt, size=(scale_list[scale_idx-1], scale_list[scale_idx-1]), mode='bilinear', align_corners=False) # (bs, 2048, 12, 12)
        img_embeds_curr_stage = img_embeds_curr_stage.permute(0, 2, 3, 1).view(b, scale_list[scale_idx-1]**2, c) # (bs, 144,  2048)
        # 插值后的拼接在输入后面
        # if scale_idx == 2:
        #     inputs_embeds = torch.cat([inputs_embeds, img_embeds_curr_stage], dim=1)
        # else:
        # cache=True
        inputs_embeds = img_embeds_curr_stage  # (bs, 144,  2048)
        (b, n, c) = inputs_embeds.shape
        inputs_embeds = torch.cat([inputs_embeds, inputs_embeds], dim=1).view(b*2, n, c) # (p*2)
        # 一步推理后面的尺度
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state

        # 当前尺度图片的token数目
        cur_image_token_nums = scale_list[scale_idx-1]**2

        logits = mmgpt.gen_head(hidden_states[:, -cur_image_token_nums:, :]) # (2*p, n, 16384)
        logit_cond = logits[0::2, :, :] # (p, n, 16384)
        logit_uncond = logits[1::2, :, :] # (p, n, 16384)
    
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond) # (p, n, 16384)
        probs = torch.softmax(logits / temperature, dim=-1) # (p, n, 16384)

        (b, n, c) = probs.shape 
        probs_2d = probs.view(b*n, c) # (2*p*n, 16384)
        
        # import ipdb; ipdb.set_trace()
        if False:
            next_tokens_2d = torch.multinomial(probs_2d, num_samples=1) # (2*p*n, 1)
        else:
            next_tokens_2d = torch.argmax(probs_2d, dim=1) # (2*p*n, 1)

        generated_tokens = next_tokens_2d.view(b, n) # (2*p, n)


        generated_img_embeds = mmgpt.prepare_gen_img_embeds(generated_tokens) # (2*p, n, 2048)
        # next_tokens = torch.multinomial(probs, num_samples=1)
        # generated_tokens = next_tokens.squeeze(dim=-1)

        dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        
      
        for i in range(parallel_size):
            save_path = os.path.join(save_image_dir, "img_{}_scale{}.jpg".format(i, scale_idx))
            PIL.Image.fromarray(visual_img[i]).save(save_path)


scale_list = vl_gpt.scale_list
patch_size = 16 
image_token_num_per_image = scale_list[0]*scale_list[0]
img_size = patch_size * scale_list[0]
generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
    image_token_num_per_image=image_token_num_per_image,
    patch_size=patch_size,
    img_size= img_size,
    scale_list=scale_list
)



"""

cd  /storage/zhubin/Janus-zb 
source /storage/miniconda3/etc/profile.d/conda.sh
conda activate janus_pro

python /storage/zhubin/Janus-zb/generation_inference_ar_with_non_ar.py



tensorboard --logdir=/storage/zhubin/Janus-zb/checkpoints

tensorboard --logdir=/storage/zhubin/Janus-zb/checkpoints/stage1_2scale_192_384_sdpa_ft_attnmask_random_replace_0.7


"""