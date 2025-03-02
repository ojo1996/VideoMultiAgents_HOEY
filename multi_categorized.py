import os
import json
from collections import defaultdict

# Load the category information
with open("data/egoschema/categories.json", "r") as f:
    categories_data = json.load(f)

# Create a mapping from question UUID to its categories
question_to_categories = {}
for item in categories_data:
    question_id, uuid, question, category_ids = item
    question_to_categories[uuid] = category_ids

# Load the text and video data
with open("data/egoschema/subset_single_text.json", "r") as f:
    text_data = json.load(f)

with open("data/egoschema/subset_single_video.json", "r") as f:
    video_data = json.load(f)

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
    # Category 5 (Action Sequence Analysis) and 1 (Purpose/Goal Identification) uses text data, all others use video data
    if 5 in categories or 1 in categories:
        if uuid in text_data:
            result[uuid] = text_data[uuid]
    else:
        if uuid in video_data:
            result[uuid] = video_data[uuid]

# Save the categorized results
with open("data/egoschema/subset_multi_categorized.json", "w") as f:
    json.dump(dict(result), f, indent=4)

print(f"Processed {len(result)} questions and saved to data/egoschema/subset_multi_categorized.json")
