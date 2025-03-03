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

for item_id, item in data.items():
    if "pred" in item:
        # Count <image> tags in all response elements
        count = 0
        for response in item.get("response", []):
            # Count occurrences of <image> in this response element
            count += response.count("<image>")
        
        image_counts.append(count)

# Calculate statistics
total_items = len(image_counts)
total_images = sum(image_counts)
mean_images = total_images / total_items if total_items > 0 else 0
median_images = np.median(image_counts)
std_images = np.std(image_counts)

# Print statistics
print(f"Total items with 'pred': {total_items}")
print(f"Total <image> tags: {total_images}")
print(f"Average <image> tags per item: {mean_images:.2f}")
print(f"Median <image> tags per item: {median_images:.2f}")
print(f"Standard deviation: {std_images:.2f}")

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
