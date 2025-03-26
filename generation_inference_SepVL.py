
import torch
# from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM_SepVL, VLChatProcessor
import numpy as np
import os
import PIL.Image


from janus.models.cosmos_tokenizer.video_lib import CausalVideoTokenizer
from janus.models.cosmos_tokenizer.utils import numpy2tensor, tensor2numpy
import mediapy as media
from tqdm import tqdm 
# specify the path to the model
model_path = "/storage/jp/Janus/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer


model_path = "/storage/zhubin/Janus-MoE/checkpoints/stage1_1scale_384_flash_ft_SepVL_Cosmos_0322/videollama3_qwen2.5_2b/stage_1/checkpoint-83000"
vl_gpt  = MultiModalityCausalLM_SepVL.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# self.gen_vision_model_decoder = gen_vision_cls(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')


conversation = [
    {
        "role": "User",
        "content": "A young girl is standing in a dirt field with a few scattered leaves and a small tree in the background. She is wearing a green dress with a floral pattern and a blue bracelet on her left wrist. Her hair is short and black.",
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM_SepVL,
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

    for i in tqdm(range(image_token_num_per_image)):
        
        if i == 0:
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None, image_token_nums=0)
        else:
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None, image_token_nums=1)
        
        
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


    # dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    # dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    # dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    # visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    # visual_img[:, :, :] = dec

    # os.makedirs('useless/test', exist_ok=True)
    # for i in range(parallel_size):
    #     save_path = os.path.join('useless/test', "img_{}.jpg".format(i))
    #     PIL.Image.fromarray(visual_img[i]).save(save_path)

    # import ipdb; ipdb.set_trace()
    h, w = img_size//patch_size, img_size//patch_size
    for i in range(parallel_size):
        # generated_tokens = generated_tokens[0] # generated_tokens: (bs, seq_len)
        pred_scale1_ids = generated_tokens[i].reshape(1,1,h,w)
        reconstructed_tensor = vl_gpt.gen_vision_model_decoder.decode(pred_scale1_ids) # b, c, t, h, w
        reconstructed_tensor = reconstructed_tensor.squeeze(2) 

        # reconstructed_tensor = reconstructed_tensor.permute(1,2,0) 
        recon_image = tensor2numpy(reconstructed_tensor)[0]
        save_dir = '/storage/zhubin/Janus-MoE/useless_'; os.makedirs(save_dir, exist_ok=True)
        save_path_scale1_recon = os.path.join(save_dir, f"scale1_infer_{i}.jpg")
        media.write_image(save_path_scale1_recon, recon_image)


generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
    parallel_size=16
)



"""

cd  /storage/zhubin/Janus-MoE/
source /storage/miniconda3/etc/profile.d/conda.sh
conda activate janus_pro

python generation_inference_SepVL.py


tensorboard --logdir=/storage/zhubin/Janus-zb/checkpoints/stage1_2scale_384_384_768_sdpa_ft_attnmask_random_replace_0.5_repeat_maskarimage

tensorboard --logdir=/storage/zhubin/Janus-MoE/checkpoints/stage1_1scale_384_flash_ft_SepVL_Cosmos_0322


"""