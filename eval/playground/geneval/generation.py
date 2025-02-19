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
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


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
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--model_path",
        type=str,
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
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
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
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)],chunk_size


def get_chunk(lst, n, k):
    chunks,chunk_size = split_list(lst, n)
    return chunks[k],chunk_size

def main(opt):
    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    metadatas,chunk_size = get_chunk(metadatas, opt.num_chunks, opt.chunk_idx)
    offset=chunk_size*opt.chunk_idx

    model_path = opt.model_path
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    

    for index in tqdm(range(len(metadatas))):
        metadata=metadatas[index]
        outpath = os.path.join(opt.outdir, f"{offset+index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
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
        n_rows = 1
        #print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0

        with torch.no_grad():
            # Generate images
            samples=generate(vl_gpt,
                vl_chat_processor,
                prompt,
                parallel_size=opt.n_samples)
            for i in range(opt.n_samples):
                Image.fromarray(samples[i]).save(os.path.join(sample_path, f"{sample_count:05}.png"))
                sample_count += 1
    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
