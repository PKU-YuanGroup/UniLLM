import os
import json
import numpy as np
import torch
from PIL import Image
import random

from tqdm.auto import tqdm
import concurrent.futures


num_segments = 1

# root directory of evaluation dimension 10
dimension10_dir = "eval/pg/seed_bench/SEED-Bench-video-image/v1_video/task10/ssv2_8_frame"
# root directory of evaluation dimension 11
dimension11_dir = "eval/pg/seed_bench/SEED-Bench-video-image/v1_video/task11/kitchen_8_frame"
# root directory of evaluation dimension 12
dimension12_dir = "eval/pg/seed_bench/SEED-Bench-video-image/v1_video/task12/breakfast_8_frame"




def fetch_images_parallel(qa_item):
    return qa_item, fetch_images(qa_item)

if __name__ == "__main__":
    data = json.load(open('eval/pg/seed_bench/SEED-Bench.json'))
    video_img_dir = 'eval/pg/seed_bench/SEED-Bench-video-image'
    ques_type_id_to_name = {id:n for n,id in data['question_type'].items()}

    video_data = [x for x in data['questions'] if x['data_type'] == 'video']
    
    load_dir=None
    for qa_item in video_data:
        img_file = f"{qa_item['question_type_id']}_{qa_item['question_id']}.png"
        if qa_item['question_type_id']==10:
            load_dir=dimension10_dir
        elif qa_item['question_type_id']==11:
            load_dir=dimension11_dir
        elif qa_item['question_type_id']==12:
            load_dir=dimension12_dir
        img=Image.open(os.path.join(load_dir,qa_item['question_id'],"1.png"))
        img.save(os.path.join(video_img_dir, img_file))
    
