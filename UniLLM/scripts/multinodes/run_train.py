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
    output_log_path = os.path.join("/storage/jp/Janus/scripts/multinodes/logs", f"output_{node}.log")
    result = subprocess.run(["ssh", f"{node}", f"nohup bash /storage/jp/Janus/scripts/train/stage1_unillm_multinodes.sh  {world_size} {rank_idx} {main_ip}> {str(output_log_path)} 2>&1 &"], capture_output=True, text=True)
    rank_idx += 1