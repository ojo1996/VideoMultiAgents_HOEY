import json
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter

# Load the JSON file
with open('data/egoschema/subset_dynamic_sampling.json', 'r') as f:
    data = json.load(f)

# Count <image> tags in responses for items with "pred"
image_counts = []
image_counts_after_first = []  # Count images after first question

for item_id, item in data.items():
    if "pred" in item:
        # Count <image> tags in all response elements
        count = 0
        count_after_first = 0
        found_first_question = False
        
        for response in item.get("response", []):
            # Count occurrences of <image> in this response element
            count += response.count("<image>")
            
            # Check if this is or contains the first question
            if not found_first_question and "[Question and 5 Options to Solve]" in response:
                found_first_question = True
                continue  # Skip counting this one in the after_first count
            
            # Count images after the first question
            if found_first_question:
                count_after_first += response.count("<image>")
        
        image_counts.append(count)
        if found_first_question:
            image_counts_after_first.append(count_after_first)

# Calculate statistics for all images
total_items = len(image_counts)
total_images = sum(image_counts)
mean_images = total_images / total_items if total_items > 0 else 0
median_images = np.median(image_counts)
std_images = np.std(image_counts)

# Calculate statistics for images after first question
total_items_after = len(image_counts_after_first)
total_images_after = sum(image_counts_after_first)
mean_images_after = total_images_after / total_items_after if total_items_after > 0 else 0
median_images_after = np.median(image_counts_after_first)
std_images_after = np.std(image_counts_after_first)

# Print statistics
print(f"Total items with 'pred': {total_items}")
print(f"Total <image> tags: {total_images}")
print(f"Average <image> tags per item: {mean_images:.2f}")
print(f"Median <image> tags per item: {median_images:.2f}")
print(f"Standard deviation: {std_images:.2f}")

print("\nStatistics for images after first question:")
print(f"Total items with question: {total_items_after}")
print(f"Total <image> tags after first question: {total_images_after}")
print(f"Average <image> tags after first question: {mean_images_after:.2f}")
print(f"Median <image> tags after first question: {median_images_after:.2f}")
print(f"Standard deviation after first question: {std_images_after:.2f}")

# Plot distribution
plt.figure(figsize=(10, 6))

# Create a counter for the image counts
count_distribution = Counter(image_counts)
counts = sorted(count_distribution.keys())
frequencies = [count_distribution[count] for count in counts]

# Calculate percentages
total_count = sum(frequencies)
percentages = [100 * freq / total_count for freq in frequencies]

# Bar plot
plt.bar(counts, percentages, color='skyblue', alpha=0.7)

# Add mean and median lines
plt.axvline(x=mean_images, color='red', linestyle='--', label=f'Mean: {mean_images:.2f}')
plt.axvline(x=median_images, color='green', linestyle='--', label=f'Median: {median_images:.2f}')

# Add labels and title
plt.xlabel('Number of Dynamic Sampling Rounds')
plt.ylabel('Percentage (%)')
plt.title('Distribution of Dynamic Sampling Rounds')
plt.xticks(counts)
plt.grid(axis='y', alpha=0.3)
plt.legend()

# Save the plot
plt.savefig('dynamic_sampling_rounds.png')
plt.show()
