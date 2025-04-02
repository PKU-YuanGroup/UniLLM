import torch
from janus.models.diffloss import DiffLoss


z = torch.rand(2,276, 57).to(torch.bfloat16)
target = torch.rand(2,276, 578).to(torch.bfloat16)
# mask = torch.rand(2,276, 578)
mask = None

diffusion_batch_mul = 4
diffloss = DiffLoss(
            target_channels=578,
            z_channels=57,
            width=578,
            depth=2,
            num_sampling_steps="50",
            grad_checkpointing=False
        )

diffloss.net = diffloss.net.to(target.dtype)
def forward_diff_loss( z, target, mask):
    bsz, seq_len, _ = target.shape
    target = target.reshape(bsz * seq_len, -1).repeat(diffusion_batch_mul, 1)
    z = z.reshape(bsz*seq_len, -1).repeat(diffusion_batch_mul, 1)
    mask = None
    loss = diffloss(z=z, target=target, mask=mask)
    return loss

forward_diff_loss( z, target, mask)



"""
 
cd /mnt/workspace/zhubin/Janus-MoE
python test_diff_loss.py

# http
git config --global http.https://github.com.proxy http://127.0.0.1:7895
git config --global https.https://github.com.proxy http://127.0.0.1:7895

git config --global --unset http.proxy
git config --global --unset https.proxy


"""