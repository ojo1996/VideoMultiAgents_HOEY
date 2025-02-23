import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.ticker as mtick
from collections import Counter

# Load JSON files
files = {
    'graph_vs_text': 'single_graph_better_than_single_text.json',
    'graph_vs_video': 'single_graph_better_than_single_video.json',
    'text_vs_graph': 'single_text_better_than_single_graph.json',
    'video_vs_text': 'single_video_better_than_single_text.json',
    'text_vs_video': 'single_text_better_than_single_video.json',
    'video_vs_graph': 'single_video_better_than_single_graph.json'
}

# Initialize counters
comparisons = defaultdict(lambda: defaultdict(int))
type_comparisons = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# Process each file
for comparison, filename in files.items():
    with open(filename, 'r') as f:
        data = json.load(f)
    
    for qid, question in data.items():
        truth = question['truth']
        q_type = question['type']
        first_modality = comparison.split('_vs_')[0]
        second_modality = comparison.split('_vs_')[1]
        
        # Get predictions
        first_pred = question[f'single_{first_modality}']['pred']
        second_pred = question[f'single_{second_modality}']['pred']
        
        # Count advantages
        if first_pred == truth and second_pred != truth:
            comparisons[first_modality][second_modality] += 1
            type_comparisons[q_type][first_modality][second_modality] += 1
        elif second_pred == truth and first_pred != truth:
            comparisons[second_modality][first_modality] += 1
            type_comparisons[q_type][second_modality][first_modality] += 1


# Initialize accuracy counters
accuracy_counts = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
majority_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
multi_star_counts = defaultdict(lambda: {'correct': 0, 'total': 0})

# Load all modality data and find intersection of answered questions
modality_data = {}
all_questions = set()
for modality in ['text', 'graph', 'video']:
    filename = f'data/nextqa/val_single_{modality}.json'
    with open(filename, 'r') as f:
        data = json.load(f)
    # Get only questions with predictions
    answered_questions = {qid: q for qid, q in data.items() if 'pred' in q}
    modality_data[modality] = answered_questions
    if not all_questions:
        all_questions = set(answered_questions.keys())
    else:
        all_questions.intersection_update(answered_questions.keys())

# Verify we have exactly 1000 common questions
assert len(all_questions) == 1000, f"Expected 1000 common questions, got {len(all_questions)}"

# Load multi-star data
with open('data/nextqa/val_multi_star_all.json', 'r') as f:
    multi_star_data = json.load(f)

# Process only the common questions
for qid in all_questions:
    q_type = modality_data['text'][qid]['type']  # Type is same across modalities
    truth = modality_data['text'][qid]['truth']  # Truth is same across modalities
    
    # Get all predictions for majority vote
    predictions = [modality_data[modality][qid]['pred'] for modality in ['text', 'graph', 'video']]
    majority_pred = Counter(predictions).most_common(1)[0][0]
    
    # Update majority counts
    majority_counts[q_type]['total'] += 1
    if majority_pred == truth:
        majority_counts[q_type]['correct'] += 1
    
    # Update multi-star counts
    if qid in multi_star_data:
        multi_star_counts[q_type]['total'] += 1
        if multi_star_data[qid]['pred'] == truth:
            multi_star_counts[q_type]['correct'] += 1
    
    for modality in ['text', 'graph', 'video']:
        pred = modality_data[modality][qid]['pred']
        
        # Update counts
        accuracy_counts[q_type][modality]['total'] += 1
        if pred == truth:
            accuracy_counts[q_type][modality]['correct'] += 1

# Calculate overall accuracies
overall_accuracies = {}
for modality in ['text', 'graph', 'video']:
    correct = sum(accuracy_counts[q_type][modality]['correct'] for q_type in accuracy_counts)
    total = sum(accuracy_counts[q_type][modality]['total'] for q_type in accuracy_counts)
    overall_accuracies[modality] = correct / total if total > 0 else 0

# Calculate overall majority accuracy
overall_majority_correct = sum(majority_counts[q_type]['correct'] for q_type in majority_counts)
overall_majority_total = sum(majority_counts[q_type]['total'] for q_type in majority_counts)
overall_accuracies['majority_vote'] = overall_majority_correct / overall_majority_total if overall_majority_total > 0 else 0

# Calculate overall multi-star accuracy
overall_multi_star_correct = sum(multi_star_counts[q_type]['correct'] for q_type in multi_star_counts)
overall_multi_star_total = sum(multi_star_counts[q_type]['total'] for q_type in multi_star_counts)
overall_accuracies['multi_star'] = overall_multi_star_correct / overall_multi_star_total if overall_multi_star_total > 0 else 0

# Print overall accuracies sorted by descending accuracy
print("\nOverall Accuracies:")
sorted_overall = sorted(overall_accuracies.items(), key=lambda x: x[1], reverse=True)
for modality, acc in sorted_overall:
    print(f"{modality}: {acc:.2%}")

# Calculate and print accuracies by question type, sorted by descending accuracy
print("\nAccuracies by Question Type:")
# Sort question types by number of questions in descending order
sorted_q_types = sorted(accuracy_counts.keys(), 
                       key=lambda x: accuracy_counts[x]['text']['total'], 
                       reverse=True)
for q_type in sorted_q_types:
    total_questions = accuracy_counts[q_type]['text']['total']
    print(f"\n{q_type} Questions ({total_questions} total):")
    # Get accuracies for this question type
    type_accuracies = []
    for modality in ['text', 'graph', 'video']:
        correct = accuracy_counts[q_type][modality]['correct']
        total = accuracy_counts[q_type][modality]['total']
        accuracy = correct / total if total > 0 else 0
        type_accuracies.append((modality, accuracy, correct, total))
    
    # Add majority vote accuracy
    correct = majority_counts[q_type]['correct']
    total = majority_counts[q_type]['total']
    accuracy = correct / total if total > 0 else 0
    type_accuracies.append(('majority_vote', accuracy, correct, total))
    
    # Add multi-star accuracy
    correct = multi_star_counts[q_type]['correct']
    total = multi_star_counts[q_type]['total']
    accuracy = correct / total if total > 0 else 0
    type_accuracies.append(('multi_star', accuracy, correct, total))
    
    # Sort by accuracy in descending order
    type_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    # Print sorted results
    for modality, accuracy, correct, total in type_accuracies:
        print(f"{modality}: {accuracy:.2%} ({correct}/{total})")


# Create figure and axis
plt.figure(figsize=(14, 8))
ax = plt.gca()

# Prepare data for plotting
x_labels = []
x_positions = []
modality_accuracies = {'text': [], 'graph': [], 'video': [], 'majority_vote': [], 'multi_star': []}

# Add overall accuracy first
x_labels.append('Overall')
x_positions.append(0)
for modality in ['text', 'graph', 'video', 'majority_vote', 'multi_star']:
    modality_accuracies[modality].append(overall_accuracies[modality])

# Add question types
for q_type in sorted_q_types:
    x_labels.append(f"{q_type}\n({accuracy_counts[q_type]['text']['total']})")
    x_positions.append(len(x_positions))
    for modality in ['text', 'graph', 'video']:
        correct = accuracy_counts[q_type][modality]['correct']
        total = accuracy_counts[q_type][modality]['total']
        accuracy = correct / total if total > 0 else 0
        modality_accuracies[modality].append(accuracy)
    # Add majority vote accuracy
    correct = majority_counts[q_type]['correct']
    total = majority_counts[q_type]['total']
    accuracy = correct / total if total > 0 else 0
    modality_accuracies['majority_vote'].append(accuracy)
    # Add multi-star accuracy
    correct = multi_star_counts[q_type]['correct']
    total = multi_star_counts[q_type]['total']
    accuracy = correct / total if total > 0 else 0
    modality_accuracies['multi_star'].append(accuracy)

# Set up bar positions and width
bar_width = 0.15
x = np.arange(len(x_labels))

# Plot bars for each modality
modalities = ['text', 'graph', 'video', 'majority_vote', 'multi_star']
for i, modality in enumerate(modalities):
    ax.bar(x + i*bar_width, modality_accuracies[modality], 
           width=bar_width, label='Majority Vote' if modality == 'majority_vote' else ('Multi-Star' if modality == 'multi_star' else modality.capitalize()))

# Add labels and title
ax.set_xlabel('Question Type (Total Questions)', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Accuracy by Question Type and Modality', fontsize=16, pad=20)
ax.set_xticks(x + bar_width * 2)
ax.set_xticklabels(x_labels, fontsize=12)
ax.legend(fontsize=12)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Add grid for better readability
ax.grid(True, axis='y', alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('modality_advantages_bar_chart.png', dpi=300)
plt.close()


def create_heatmap(data, title, filename):
    """Helper function to create and save a heatmap"""
    modalities = ['text', 'graph', 'video']
    data_matrix = [[data[m1][m2] for m2 in modalities] for m1 in modalities]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_matrix, cmap='YlGn')

    ax.set_xticks(range(len(modalities)))
    ax.set_yticks(range(len(modalities)))
    ax.set_xticklabels(modalities, fontsize=14)
    ax.set_yticklabels(modalities, fontsize=14)
    ax.set_xlabel('Missed by', fontsize=16)
    ax.set_ylabel('Correct in', fontsize=16)
    plt.title(title, fontsize=18, pad=20)

    for i in range(len(modalities)):
        for j in range(len(modalities)):
            text_color = 'white' if data_matrix[i][j] > (max(map(max, data_matrix)) / 2) else 'black'
            text = ax.text(j, i, data_matrix[i][j],
                          ha="center", va="center", color=text_color, fontsize=16)

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# Create overall heatmap
create_heatmap(comparisons, 
              'Advantages of Each Modality in NextQA Validation',
              'modality_advantages.png')

# Create heatmaps for each question type and print accuracy rankings
for q_type, type_data in type_comparisons.items():
    create_heatmap(type_data,
                  f'Advantages by Modality for {q_type} Questions',
                  f'modality_advantages_{q_type}.png')
