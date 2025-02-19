import json
import os
input_file = '/storage/mjc/Janus/eval/pg/seed_bench/SEED-Bench.json'
output_file = '/storage/mjc/Janus/eval/pg/seed_bench/new-seed-bench-video.json'
data = json.load(open(input_file))
ques_type_id_to_name = {id:n for n,id in data['question_type'].items()}
video_data = [x for x in data['questions'] if x['data_type'] == 'video']
print(len(video_data),len(data['questions']))
with open(output_file, 'w') as outfile:
    for data in video_data:
        newdata={}
        if data["data_type"]=="image":
            continue
        newdata["image"]=os.path.join("SEED-Bench-video-image", f"{data['question_type_id']}_{data['question_id']}.png")
        newdata["text"]=data["question"]+"\nA. "+data["choice_a"]+"B. \n"+data["choice_b"]+"\nC. "+data["choice_c"]+"\nD. "+data["choice_d"]+"\n"+"Answer with the option's letter from the given choices directly."
        newdata["category"]=ques_type_id_to_name[data['question_type_id']]
        newdata["question_id"]=data['question_id']
        outfile.write(json.dumps(newdata) + '\n')