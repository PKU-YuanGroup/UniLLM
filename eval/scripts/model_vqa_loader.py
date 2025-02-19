import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder,processor):
        self.questions = questions
        self.image_folder = image_folder
        self.processor=processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = os.path.join(self.image_folder,line["image"])
        qs = line["text"]

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{qs}",
                "images": [image_file],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation, images=pil_images, force_batchify=True
        )
        #exit()
        # # run image encoder to get the image embeddings
        return prepare_inputs

    def __len__(self):
        return len(self.questions)
def simple_collate_fn(batch):
    return batch[0]

# DataLoader
def create_data_loader(questions, image_folder, processor, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,collate_fn=simple_collate_fn, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    data_loader = create_data_loader(questions, args.image_folder,vl_chat_processor)
    cnt = -1
    for prepare_inputs, line in tqdm(zip(data_loader, questions), total=len(questions)):
        cnt += 1
        # if cnt == 30:
        #     break
        idx = line["question_id"]
        cur_prompt = line["text"]

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
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": args.model_path,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="deepseek-ai/Janus-Pro-7B")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    eval_model(args)
