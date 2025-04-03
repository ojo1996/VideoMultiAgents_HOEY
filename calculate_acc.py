import os
import json

# Configuration
RESULT_FILE_PATH   = "subset_anno.json"

# Load result_data
with open(RESULT_FILE_PATH, "r") as f:
    result_data = json.load(f)

items_num   = 0
correct_num = 0

# Loop through result_data
for i, (video_id, json_data) in enumerate(result_data.items()):
    if "pred" in json_data.keys() and json_data["pred"] != -2:
        items_num += 1
    try:
        if json_data["pred"] == json_data["truth"]:
            correct_num += 1
    except:
        pass
print ("------------------------------------")
print("Items: {}".format(items_num))
print("Correct: {}".format(correct_num))
print("Accuracy: {:.2f}%".format(correct_num / items_num * 100))