import subprocess
import sys
import os
sys.path.append(".")
from  config import nodes

for node in nodes:
    print(node)
    result = subprocess.run(["ssh", f"{node}", f"pkill -f /train.py"], capture_output=True, text=True)
    result = subprocess.run(["ssh", f"{node}", f"pkill -f /train_var.py"], capture_output=True, text=True)
    result = subprocess.run(["ssh", f"{node}", f"pkill -f  v2.py"], capture_output=True, text=True)


"""
ps -ef | grep objaverse | grep -v grep | awk '{print $2}' | xargs kill -9
pkill -f -9 step1_get_bbox_gpu.py
"""