"""Adapted from TODO"""

import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from transformers import AutoModelForCausalLM,AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor

from janus.models.modeling_vlm import MultiModalityConfig
@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    return dec

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument(
        "--prompt_dirs",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=384,
        help="resolution",
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1)
    opt = parser.parse_args()
    return opt
import math
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
from janus.models.modeling_vlm import MultiModalityConfig
def main(opt):
    # Load prompts
    prompts_files=os.listdir(opt.prompt_dirs)
    prompts_files = get_chunk(prompts_files, opt.num_chunks, opt.chunk_idx)
    model_path = opt.model_path
    config = MultiModalityConfig.from_pretrained(model_path, cache_dir='./cache_dir')
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, config=config,  cache_dir='./cache_dir'
    )
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained("/storage/jp/Janus/Janus-Pro-1B")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vl_chat_processor.tokenizer=tokenizer
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    os.makedirs(opt.outdir, exist_ok=True)
    for prompt_file in tqdm(prompts_files):
        with open(os.path.join(opt.prompt_dirs, prompt_file)) as f:
            prompt = f.readlines()[0].strip()
        outpath = os.path.join(opt.outdir, prompt_file.split(".txt")[0]+".png")

        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag
        with torch.no_grad():
            # Generate images
            samples=generate(vl_gpt,
                vl_chat_processor,
                prompt,
                parallel_size=opt.n_samples,
                img_size=opt.size).to(torch.float32)
            grid = make_grid(samples, nrow=2)
            
            # to image
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = np.clip((grid + 1) / 2 * 255, 0, 255)
            grid = Image.fromarray(grid.astype(np.uint8))
            grid.save(outpath)
            del grid
    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
