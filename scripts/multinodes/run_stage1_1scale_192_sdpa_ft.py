import subprocess
import sys
import os
sys.path.append(".")
from  config import nodes



rank_idx = 0
world_size=len(nodes)
main_ip=nodes[0]
for node in nodes[:]:
    print(rank_idx)

    log_dir = "/storage/zhubin/Janus-zb/logs/logs_stage1_1scale_192_sdpa_ft"
    os.makedirs(log_dir, exist_ok=True)
    output_log_path = os.path.join(log_dir, f"output_{node}.log")
    result = subprocess.run(["ssh", f"{node}", f"nohup bash /storage/zhubin/Janus-zb/scripts/multinodes/stage1_1scale_192_sdpa_ft.sh  {world_size} {rank_idx} {main_ip}> {str(output_log_path)} 2>&1 &"], capture_output=True, text=True)
    rank_idx += 1

    