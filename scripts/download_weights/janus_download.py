
from huggingface_hub import login, snapshot_download
import os

login(token="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", add_to_git_credential=True)
model_names = [
        # "Cosmos-0.1-Tokenizer-CI8x8",
        # "Cosmos-0.1-Tokenizer-CI16x16",
        # "Cosmos-0.1-Tokenizer-CV4x8x8",
        # "Cosmos-0.1-Tokenizer-CV8x8x8",
        # "Cosmos-0.1-Tokenizer-CV8x16x16",
        # "Cosmos-0.1-Tokenizer-DI8x8",
        # "Cosmos-0.1-Tokenizer-DI16x16",
        # "Cosmos-0.1-Tokenizer-DV4x8x8",
        # "Cosmos-0.1-Tokenizer-DV8x8x8",
        # "Cosmos-1.0-Tokenizer-CV8x8x8"
        "Janus-Pro-7B",
        # "Cosmos-1.0-Tokenizer-CV8x8x8",
        # "Cosmos-1.0-Tokenizer-DV8x16x16",
]
for model_name in model_names:
    hf_repo = "deepseek-ai/" + model_name  
#     fd_repo = 'nvidia/Cosmos-1.0-Tokenizer-CV8x8x8'
    local_dir = "/storage/zhubin/Janus-MoE/pretrained_ckpts/" + model_name
    os.makedirs(local_dir, exist_ok=True)
    print(f"downloading {model_name}...")
    snapshot_download(repo_id=hf_repo, local_dir=local_dir)

# 