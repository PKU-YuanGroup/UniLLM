import torch
from janus.models.diffloss import DiffLoss


z = torch.rand(2,276, 578)
target = torch.rand(2,276, 578)
# mask = torch.rand(2,276, 578)
mask = None

diffusion_batch_mul = 4
diffloss = DiffLoss(
            target_channels=578,
            z_channels=578,
            width=578,
            depth=2,
            num_sampling_steps=50,
            grad_checkpointing=False
        )
def forward_diff_loss( z, target, mask):
    bsz, seq_len, _ = target.shape
    target = target.reshape(bsz * seq_len, -1).repeat(diffusion_batch_mul, 1)
    z = z.reshape(bsz*seq_len, -1).repeat(diffusion_batch_mul, 1)
    mask = None
    loss = diffloss(z=z, target=target, mask=mask)
    return loss

forward_diff_loss( z, target, mask)

"""
conda activate janus_pro
cd /storage/zhubin/Janus-MoE/
python /storage/zhubin/Janus-MoE/test_diff_loss.py
"""