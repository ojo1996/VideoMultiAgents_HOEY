# Script to convert the depth_expansion_res.json to the final format used in MAS

import json

depth_expansion_res = "/path/to/depth_expansion_res.json"   # Final output of original VideoTree method
save_path = "/path/to/save_folder/egoschema_videotree_result.json"

with open(depth_expansion_res, "r") as f:
    depth_expansion_res = json.load(f)

result_dict = {}

for item in depth_expansion_res:
    key = item["name"]
    result_dict[key] = {
        "sorted_values": item["sorted_values"],
        "relevance": item["relevance"]
    }


with open(save_path, "w") as f:
    json.dump(result_dict, f)