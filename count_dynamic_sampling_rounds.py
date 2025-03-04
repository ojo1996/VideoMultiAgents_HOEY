import json
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
import datetime
from scipy.stats import pearsonr

# Load the JSON file
with open('data/egoschema/subset_dynamic_sampling_gpt-4o-2024-08-06_max_iter_10.json', 'r') as f:
    data = json.load(f)

# Count <image> tags in responses for items with "pred"
image_counts = []
image_counts_after_first = []  # Count images after first question
num_frames_counts = []  # Count num_frames in more_frames decisions
start_timestamps = []  # Store start_timestamp values
end_timestamps = []    # Store end_timestamp values
correct_predictions = []  # Store whether prediction was correct
item_ids = []  # Store item IDs for reference
timestamp_frames_tuples = []  # Store (start_timestamp, end_timestamp, num_frames) tuples

for item_id, item in data.items():
    if "pred" in item:
        # Count <image> tags in all response elements
        count = 0
        count_after_first = 0
        found_first_question = False
        
        for response in item.get("response", []):
            # Count occurrences of <image> in this response element
            count += response.count("PngImageFile")
            
            # Check if this is or contains the first question
            if not found_first_question and "[Question and 5 Options to Solve]" in response:
                found_first_question = True
                continue  # Skip counting this one in the after_first count
            
            # Count images after the first question
            if found_first_question:
                count_after_first += response.count("PngImageFile")
            
            # Extract num_frames from responses starting with "{"
            if response.startswith("{"):
                try:
                    response_json = json.loads(response)
                    if "decision" in response_json and response_json["decision"]["type"] == "more_frames":
                        num_frames = response_json["decision"]["num_frames"]
                        num_frames_int = int(num_frames)
                        num_frames_counts.append(num_frames_int)
                        
                        # Extract timestamps if they exist
                        start_timestamp = None
                        end_timestamp = None
                        if "start_timestamp" in response_json["decision"]:
                            start_timestamp = response_json["decision"]["start_timestamp"]
                            start_timestamps.append(start_timestamp)
                        if "end_timestamp" in response_json["decision"]:
                            end_timestamp = response_json["decision"]["end_timestamp"]
                            end_timestamps.append(end_timestamp)
                        
                        # Store the tuple if we have both timestamps
                        if start_timestamp and end_timestamp:
                            timestamp_frames_tuples.append((start_timestamp, end_timestamp, num_frames_int))
                except (json.JSONDecodeError, KeyError):
                    pass
        
        image_counts.append(count)
        if found_first_question:
            image_counts_after_first.append(count_after_first)
        
        # Store whether prediction was correct (1 for correct, 0 for incorrect)
        is_correct = 1 if item.get("pred") == item.get("truth", "") else 0
        correct_predictions.append(is_correct)
        item_ids.append(item_id)

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

# Calculate statistics for num_frames
mean_num_frames = np.mean(num_frames_counts) if num_frames_counts else 0
median_num_frames = np.median(num_frames_counts) if num_frames_counts else 0
std_num_frames = np.std(num_frames_counts) if num_frames_counts else 0

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

print("\nStatistics for num_frames in more_frames decisions:")
print(f"Total num_frames decisions: {len(num_frames_counts)}")
print(f"Mean num_frames: {mean_num_frames:.2f}")
print(f"Median num_frames: {median_num_frames:.2f}")
print(f"Standard deviation of num_frames: {std_num_frames:.2f}")

# Calculate correlation between number of rounds and correctness
correlation, p_value = pearsonr(image_counts, correct_predictions)
print("\nCorrelation between number of rounds and correctness:")
print(f"Pearson correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# Plot distribution of image counts
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

# Plot scatterplot of rounds vs correctness
plt.figure(figsize=(12, 6))

# Create jittered y-values for better visualization
jitter = np.random.normal(0, 0.05, size=len(correct_predictions))
jittered_correct = np.array(correct_predictions) + jitter

# Create the scatterplot
plt.scatter(image_counts, jittered_correct, alpha=0.5, c='blue')

# Add a trend line
z = np.polyfit(image_counts, correct_predictions, 1)
p = np.poly1d(z)
plt.plot(sorted(set(image_counts)), p(sorted(set(image_counts))), "r--", 
         label=f"Trend line (r={correlation:.4f}, p={p_value:.4f})")

# Add labels and title
plt.xlabel('Number of Dynamic Sampling Rounds')
plt.ylabel('Prediction Correctness (1=Correct, 0=Incorrect)')
plt.title('Correlation Between Number of Rounds and Prediction Correctness')
plt.yticks([0, 1], ['Incorrect', 'Correct'])
plt.grid(True, alpha=0.3)
plt.legend()

# Save the plot
plt.savefig('rounds_vs_correctness.png')
plt.show()

# Plot distribution of num_frames
plt.figure(figsize=(10, 6))

# Create a counter for the num_frames counts
num_frames_distribution = Counter(num_frames_counts)
num_frames_values = sorted(num_frames_distribution.keys())
num_frames_frequencies = [num_frames_distribution[count] for count in num_frames_values]

# Calculate percentages
total_num_frames = sum(num_frames_frequencies)
num_frames_percentages = [100 * freq / total_num_frames for freq in num_frames_frequencies]

# Bar plot
plt.bar(num_frames_values, num_frames_percentages, color='lightgreen', alpha=0.7)

# Add mean and median lines
plt.axvline(x=mean_num_frames, color='red', linestyle='--', label=f'Mean: {mean_num_frames:.2f}')
plt.axvline(x=median_num_frames, color='green', linestyle='--', label=f'Median: {median_num_frames:.2f}')

# Add labels and title
plt.xlabel('Number of Frames Requested (num_frames)')
plt.ylabel('Percentage (%)')
plt.title('Distribution of num_frames in more_frames Decisions')
plt.xticks(num_frames_values)
plt.grid(axis='y', alpha=0.3)
plt.legend()

# Save the plot
plt.savefig('num_frames_distribution.png')
plt.show()

# Function to convert timestamp to seconds
def timestamp_to_seconds(timestamp):
    # Handle different timestamp formats
    if ':' in timestamp:
        parts = timestamp.split(':')
        if len(parts) == 2:  # mm:ss format
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:  # hh:mm:ss format
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(timestamp)  # Assume it's already in seconds

# Generate sampled timestamps using the stored tuples
sampled_timestamps = []
for start_ts, end_ts, num_frames in timestamp_frames_tuples:
    start_sec = timestamp_to_seconds(start_ts)
    end_sec = timestamp_to_seconds(end_ts)
    
    # Generate uniformly sampled timestamps between start and end
    if start_sec < end_sec:  # Ensure valid interval
        interval_samples = np.linspace(start_sec, end_sec, num_frames)
        sampled_timestamps.extend(interval_samples)

# Plot distribution of timestamps
plt.figure(figsize=(12, 6))

# Create histogram for sampled timestamps
bins = np.linspace(0, 180, 37)  # 5-second bins from 0 to 180 seconds (3 minutes)
sampled_hist, _ = np.histogram(sampled_timestamps, bins=bins)

# Calculate percentages
sampled_percentage = 100 * sampled_hist / len(sampled_timestamps) if len(sampled_timestamps) > 0 else sampled_hist

# Plot histogram
plt.bar(bins[:-1], sampled_percentage, width=(bins[1]-bins[0])*0.8, color='green', alpha=0.7, label='Sampled Frames')

# Add labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Percentage (%)')
plt.title('Distribution of Sampled Timestamps in Dynamic Sampling')

# Create custom x-ticks in mm:ss format
x_ticks = np.arange(0, 181, 15)  # Every 15 seconds
x_labels = [f"{int(x//60):02d}:{int(x%60):02d}" for x in x_ticks]
plt.xticks(x_ticks, x_labels)

plt.grid(axis='y', alpha=0.3)
plt.legend()

# Save the plot
plt.savefig('timestamp_distribution.png')
plt.show()
