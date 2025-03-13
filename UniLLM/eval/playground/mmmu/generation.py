import torch
import os
import random

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets

from argparse import ArgumentParser

from utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from utils.model_utils import call_llava_engine_df, llava_image_processor
from utils.eval_utils import parse_multi_choice_response, parse_open_response
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import math
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import pandas as pd
import glob,io
from PIL import Image
def main():
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, default="deepseek-ai/Janus-Pro-7B")
    parser.add_argument('--output_path', type=str, default='janus_val.json',
                        help='name of saved json')
    parser.add_argument('--prompt_config', type=str, default="configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    set_seed(args.seed)

    # load config and process to one value
    args.config = load_yaml(args.prompt_config)
    
    # run for each subject
    df_list = []
    for subject in CAT_SHORT2LONG.values():
        file_list=glob.glob(os.path.join(args.data_path,subject,f"{args.split}-*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in file_list], ignore_index=True)
        df_list.append(df)

    # merge all dataset
    dataset = pd.concat(df_list, ignore_index=True)
    dataset = get_chunk(dataset, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.output_path)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    length=len(dataset)
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    print(args.chunk_idx,args.split,length)

    results={}
    for idx in range(length):
        vid_meta=dataset.iloc[idx]
        sample = process_single_sample(vid_meta)
        prompt,res_dict = construct_prompt(sample, args.config)
        #print(sample["final_input_prompt"])
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": sample["image"],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images =[]
        for image_dict in sample["image"]:
            pil_images.append(Image.open(io.BytesIO(image_dict["bytes"])).convert('RGB'))
        if len(pil_images)>1:
            outputs = random.choice(res_dict["all_choices"])
            results[sample["id"]]=outputs
            continue
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        )
        prepare_inputs = prepare_inputs.to(vl_gpt.device)
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        outputs = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        if sample['question_type'] == 'multiple-choice':
            outputs = parse_multi_choice_response(outputs, res_dict["all_choices"], res_dict["index2ans"])
        results[sample["id"]]=outputs
    
    save_json(args.output_path, results)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()

