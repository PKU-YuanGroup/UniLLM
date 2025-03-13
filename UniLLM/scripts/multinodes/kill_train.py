import subprocess
import sys
import os
sys.path.append(".")
from  config import nodes

for node in nodes:
    print(node)
    result = subprocess.run(["ssh", f"{node}", f"pkill -f trainer/train_v2.py"], capture_output=True, text=True)