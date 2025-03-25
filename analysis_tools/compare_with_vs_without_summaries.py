import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

# Compare performance with and without video summaries
def load_results(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# Load results
with_summaries = load_results('data/nextqa/val_single_video.json')
without_summaries = load_results('data/nextqa/val_single_video_no_summaries.json')

# Calculate accuracies
def calculate_accuracies(data):
    accuracies = {'overall': {'correct': 0, 'total': 0}}
    for q_id, q_data in data.items():
        if 'pred' not in q_data:
            continue
            
        q_type = q_data['type']
        if q_type not in accuracies:
            accuracies[q_type] = {'correct': 0, 'total': 0}
            
        # Update overall counts
        accuracies['overall']['total'] += 1
        if q_data['pred'] == q_data['truth']:
            accuracies['overall']['correct'] += 1
            
        # Update question type counts    
        accuracies[q_type]['total'] += 1
        if q_data['pred'] == q_data['truth']:
            accuracies[q_type]['correct'] += 1
            
    # Calculate accuracy percentages
    for key in accuracies:
        accuracies[key]['accuracy'] = accuracies[key]['correct'] / accuracies[key]['total'] if accuracies[key]['total'] > 0 else 0
        
    return accuracies

with_summary_acc = calculate_accuracies(with_summaries)
without_summary_acc = calculate_accuracies(without_summaries)

# Create bar chart comparing performance
fig, ax = plt.subplots(figsize=(12, 6))

# Get question types (including overall)
q_types = sorted(list(with_summary_acc.keys()))

# Set up bars
x = np.arange(len(q_types))
width = 0.35

# Plot bars
with_summary_bars = ax.bar(x - width/2, 
                          [with_summary_acc[qt]['accuracy'] for qt in q_types],
                          width, label='With Summaries')
without_summary_bars = ax.bar(x + width/2, 
                             [without_summary_acc[qt]['accuracy'] for qt in q_types],
                             width, label='Without Summaries')

# Customize chart
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Performance Comparison: With vs Without Video Summaries', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f"{qt}\n({with_summary_acc[qt]['total']})" for qt in q_types], fontsize=12)
ax.legend(fontsize=12)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Add grid
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1%}',
                   xy=(rect.get_x() + rect.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', rotation=0)

autolabel(with_summary_bars)
autolabel(without_summary_bars)

plt.tight_layout()
plt.savefig('single_video_with_vs_without_summary.png', dpi=300)
plt.close()
