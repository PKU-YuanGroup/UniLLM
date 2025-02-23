import torch
from transformers import AutoModelForCausalLM
from janus.models.modeling_vlm import MultiModalityConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
# specify the path to the model
model_path = "/storage/jp/Janus/work_dirs_0218/videollama3_qwen2.5_2b/stage_1/checkpoint-12000"
config = MultiModalityConfig.from_pretrained(model_path, cache_dir='./cache_dir')
model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, config=config,  cache_dir='./cache_dir'
)
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained("/storage/jp/Janus/Janus-Pro-1B")
tokenizer = vl_chat_processor.tokenizer

vl_gpt = model.to(torch.bfloat16).cuda().eval()

question = "Describe the image."
image = "./generated_samples/img_0.jpg"

conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)
#print(prepare_inputs)
#exit()
# # run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)


# # run the model to get the response
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

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)