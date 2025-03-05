import os
import json
from collections import defaultdict

# Load the category information
with open("data/results/egoschema_fullset_categories.json", "r") as f:
    categories_data = json.load(f)

# Create a mapping from question UUID to its categories
question_to_categories = {}
for item in categories_data:
    question_id, uuid, question, category_ids = item
    question_to_categories[uuid] = category_ids

# Load the text and video data
with open("data/results/egoschema_fullset_single_text.json", "r") as f:
    text_data = json.load(f)

with open("data/results/egoschema_fullset_single_video.json", "r") as f:
    video_data = json.load(f)

with open("data/results/egoschema_fullset_single_graph.json", "r") as f:
    graph_data = json.load(f)

# Initialize the result dictionary
result = defaultdict(dict)

# Process each question in the annotation data
for video_id, data in text_data.items():
    uuid = video_id
    
    # Skip if the question is not in our categories mapping
    if uuid not in question_to_categories:
        continue
    
    categories = question_to_categories[uuid]
    
    # Determine which data source to use based on the category
    if 0 in categories:
        result[uuid] = graph_data[uuid]
    elif {1, 2, 3} & set(categories):
        if uuid in text_data:
            result[uuid] = text_data[uuid]
    else:
        if uuid in video_data:
            result[uuid] = video_data[uuid]

# Save the categorized results
with open("data/results/egoschema_fullset_multi_categorized.json", "w") as f:
    json.dump(dict(result), f, indent=4)

print(f"Processed {len(result)} questions and saved to results/egoschema_fullset_multi_categorized.json")
