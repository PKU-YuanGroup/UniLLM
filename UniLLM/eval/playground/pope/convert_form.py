import json

input_file = '/storage/mjc/Janus/eval/pg/pope/coco/coco_pope_random.json'
output_file = '/storage/mjc/Janus/eval/pg/pope/new_random.json'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        
        data['question_id'] = int(f"1000{data['question_id']:04d}")
        if 'label' in data:
            del data['label']
        data['category'] = 'random'
        outfile.write(json.dumps(data) + '\n')