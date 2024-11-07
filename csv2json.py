import json
import pandas as pd
from collections import defaultdict

val_csv_path = "/root/nas_nextqa/nextqa/val.csv"
nextqa_json_path = "/root/nas_nextqa/nextqa/nextqa.json"
df = pd.read_csv(val_csv_path)
nextqa_file = defaultdict(dict)

for id, row in df.iterrows():
    nextqa_file[str(row['video']) + '_' + str(row['qid'])]['q_uid'] = str(row['video']) + '_' + str(row['qid'])
    nextqa_file[str(row['video']) + '_' + str(row['qid'])]['type'] = row['type']
    nextqa_file[str(row['video']) + '_' + str(row['qid'])]['question'] = row['question']
    nextqa_file[str(row['video']) + '_' + str(row['qid'])]['option 0'] = row['a0']
    nextqa_file[str(row['video']) + '_' + str(row['qid'])]['option 1'] = row['a1']
    nextqa_file[str(row['video']) + '_' + str(row['qid'])]['option 2'] = row['a2']
    nextqa_file[str(row['video']) + '_' + str(row['qid'])]['option 3'] = row['a3']
    nextqa_file[str(row['video']) + '_' + str(row['qid'])]['option 4'] = row['a4']
    nextqa_file[str(row['video']) + '_' + str(row['qid'])]['truth'] = row['answer']

# print(len(nextqa_file))

with open(nextqa_json_path, "w") as f:
    json.dump(nextqa_file, f, indent=4)
