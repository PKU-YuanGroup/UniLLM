import subprocess
import sys
import os
sys.path.append(".")
from  config import nodes



rank_idx = 0
world_size=len(nodes)
main_ip=nodes[0]
mask_ratio = 0.7

for node in nodes[:]:
    print(rank_idx)
    log_dir = f"/storage/zhubin/Janus-zb/logs/logs_stage1_2scale_192_384_sdpa_ft_attnmask_random_replace_{mask_ratio}"
    os.makedirs(log_dir, exist_ok=True)
    output_log_path = os.path.join(log_dir, f"output_{node}.log")
    result = subprocess.run(["ssh", f"{node}", f"nohup bash /storage/zhubin/Janus-zb/scripts/multinodes/stage1_2scale_192_384_sdpa_ft_attnmask_random_replace.sh  {world_size} {rank_idx} {main_ip} {mask_ratio}> {str(output_log_path)} 2>&1 &"], capture_output=True, text=True)
    rank_idx += 1

