import json
import glob
from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument("--answer_path", type=str)
parser.add_argument("--split", type=str, )
args = parser.parse_args()
merged_data = {}
for file in glob.glob(os.path.join(args.answer_path,f"{args.split}_*.jsonl")):
    with open(file, "r") as f:
        data = json.load(f)
        merged_data.update(data)

with open(os.path.join(args.answer_path,f"{args.split}_merge.jsonl"), "w") as f:
    json.dump(merged_data, f, indent=2)