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
    if modality == 'text':
        # m = 'text_llava1.5_no_summaries'
        # m = 'text_no_summaries'
        m = 'text_with_summaries'
    elif modality == 'graph':
        # m = 'graph_no_summaries'
        m = 'graph_gpt4o_with_summaries'
    elif modality == 'video':
        m = 'video_gemini_with_summaries'

    # m = modality

    filename = f'data/nextqa/val_single_{m}.json'
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
# assert len(all_questions) == 1000, f"Expected 1000 common questions, got {len(all_questions)}"

# Load multi-star data
with open('data/nextqa/val_multi_star_all_gemini_with_summaries.json', 'r') as f:
    multi_star_data = json.load(f)

# Process only the common questions
for qid in all_questions:
    q_type = modality_data['text'][qid]['type']  # Type is same across modalities
    truth = modality_data['text'][qid]['truth']  # Truth is same across modalities
    
    # Get all predictions for majority vote
    predictions = [modality_data[modality][qid]['pred'] for modality in ['text', 'video', 'graph']]
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


"""
Calculate agreement distribution by question type
"""
# Calculate agreement counts
def calculate_agreement_counts():
    agreement_counts = defaultdict(lambda: {
        'all_agree': {'correct': 0, 'incorrect': 0},
        'two_agree': {
            'video_text': {'correct': 0, 'incorrect': 0},
            'text_graph': {'correct': 0, 'incorrect': 0},
            'graph_video': {'correct': 0, 'incorrect': 0}
        }, 
        'all_disagree': {
            'video_correct': 0,
            'text_correct': 0,
            'graph_correct': 0,
            'none_correct': 0
        },
        'total': 0
    })
    
    # Process each question
    for qid in all_questions:
        q_type = modality_data['text'][qid]['type']
        predictions = {
            modality: modality_data[modality][qid]['pred'] 
            for modality in ['text', 'video', 'graph']
        }
        ground_truth = modality_data['text'][qid]['truth']
        
        # Count unique predictions
        unique_preds = len(set(predictions.values()))
        
        # Update counts
        agreement_counts[q_type]['total'] += 1
        if unique_preds == 1:  # All agree
            # Check if the agreed prediction is correct
            if list(predictions.values())[0] == ground_truth:
                agreement_counts[q_type]['all_agree']['correct'] += 1
            else:
                agreement_counts[q_type]['all_agree']['incorrect'] += 1
        elif unique_preds == 2:  # Two agree
            # Determine which two modalities agree
            if predictions['video'] == predictions['text'] != predictions['graph']:
                majority_pred = predictions['video']  # Same as text
                if majority_pred == ground_truth:
                    agreement_counts[q_type]['two_agree']['video_text']['correct'] += 1
                else:
                    agreement_counts[q_type]['two_agree']['video_text']['incorrect'] += 1
            elif predictions['text'] == predictions['graph'] != predictions['video']:
                majority_pred = predictions['text']  # Same as graph
                if majority_pred == ground_truth:
                    agreement_counts[q_type]['two_agree']['text_graph']['correct'] += 1
                else:
                    agreement_counts[q_type]['two_agree']['text_graph']['incorrect'] += 1
            elif predictions['graph'] == predictions['video'] != predictions['text']:
                majority_pred = predictions['graph']  # Same as video
                if majority_pred == ground_truth:
                    agreement_counts[q_type]['two_agree']['graph_video']['correct'] += 1
                else:
                    agreement_counts[q_type]['two_agree']['graph_video']['incorrect'] += 1
        else:  # All disagree
            # Check which modality (if any) got it right
            if predictions['video'] == ground_truth:
                agreement_counts[q_type]['all_disagree']['video_correct'] += 1
            elif predictions['text'] == ground_truth:
                agreement_counts[q_type]['all_disagree']['text_correct'] += 1
            elif predictions['graph'] == ground_truth:
                agreement_counts[q_type]['all_disagree']['graph_correct'] += 1
            else:
                agreement_counts[q_type]['all_disagree']['none_correct'] += 1
            
    return agreement_counts

# Calculate agreement percentages
agreement_counts = calculate_agreement_counts()

# Calculate overall percentages
overall_counts = {
    'all_agree': {'correct': 0, 'incorrect': 0},
    'two_agree': {
        'video_text': {'correct': 0, 'incorrect': 0},
        'text_graph': {'correct': 0, 'incorrect': 0},
        'graph_video': {'correct': 0, 'incorrect': 0}
    }, 
    'all_disagree': {
        'video_correct': 0,
        'text_correct': 0,
        'graph_correct': 0,
        'none_correct': 0
    },
    'total': 0
}

for q_type in agreement_counts:
    overall_counts['all_agree']['correct'] += agreement_counts[q_type]['all_agree']['correct']
    overall_counts['all_agree']['incorrect'] += agreement_counts[q_type]['all_agree']['incorrect']
    overall_counts['total'] += agreement_counts[q_type]['total']
    for pair in ['video_text', 'text_graph', 'graph_video']:
        overall_counts['two_agree'][pair]['correct'] += agreement_counts[q_type]['two_agree'][pair]['correct']
        overall_counts['two_agree'][pair]['incorrect'] += agreement_counts[q_type]['two_agree'][pair]['incorrect']
    for key in ['video_correct', 'text_correct', 'graph_correct', 'none_correct']:
        overall_counts['all_disagree'][key] += agreement_counts[q_type]['all_disagree'][key]

# Set up the plot
plt.figure(figsize=(15, 8))

# Prepare data for plotting
x = np.arange(len(sorted_q_types) + 1)  # +1 for overall
width = 0.25

# Calculate percentages for overall and each question type
all_agree_correct_percentages = []
all_agree_incorrect_percentages = []
video_text_correct_percentages = []
video_text_incorrect_percentages = []
text_graph_correct_percentages = []
text_graph_incorrect_percentages = []
graph_video_correct_percentages = []
graph_video_incorrect_percentages = []
all_disagree_video_correct_percentages = []
all_disagree_text_correct_percentages = []
all_disagree_graph_correct_percentages = []
all_disagree_none_correct_percentages = []
labels = ['Overall'] + sorted_q_types

for label in labels:
    counts = overall_counts if label == 'Overall' else agreement_counts[label]
    total = counts['total']
    if total > 0:
        all_agree_correct_pct = (counts['all_agree']['correct'] / total) * 100
        all_agree_incorrect_pct = (counts['all_agree']['incorrect'] / total) * 100
        
        video_text_correct_pct = (counts['two_agree']['video_text']['correct'] / total) * 100
        video_text_incorrect_pct = (counts['two_agree']['video_text']['incorrect'] / total) * 100
        
        text_graph_correct_pct = (counts['two_agree']['text_graph']['correct'] / total) * 100
        text_graph_incorrect_pct = (counts['two_agree']['text_graph']['incorrect'] / total) * 100
        
        graph_video_correct_pct = (counts['two_agree']['graph_video']['correct'] / total) * 100
        graph_video_incorrect_pct = (counts['two_agree']['graph_video']['incorrect'] / total) * 100
        
        all_disagree_video_correct_pct = (counts['all_disagree']['video_correct'] / total) * 100
        all_disagree_text_correct_pct = (counts['all_disagree']['text_correct'] / total) * 100
        all_disagree_graph_correct_pct = (counts['all_disagree']['graph_correct'] / total) * 100
        all_disagree_none_correct_pct = (counts['all_disagree']['none_correct'] / total) * 100
        
        all_agree_correct_percentages.append(all_agree_correct_pct)
        all_agree_incorrect_percentages.append(all_agree_incorrect_pct)
        video_text_correct_percentages.append(video_text_correct_pct)
        video_text_incorrect_percentages.append(video_text_incorrect_pct)
        text_graph_correct_percentages.append(text_graph_correct_pct)
        text_graph_incorrect_percentages.append(text_graph_incorrect_pct)
        graph_video_correct_percentages.append(graph_video_correct_pct)
        graph_video_incorrect_percentages.append(graph_video_incorrect_pct)
        all_disagree_video_correct_percentages.append(all_disagree_video_correct_pct)
        all_disagree_text_correct_percentages.append(all_disagree_text_correct_pct)
        all_disagree_graph_correct_percentages.append(all_disagree_graph_correct_pct)
        all_disagree_none_correct_percentages.append(all_disagree_none_correct_pct)
    else:
        all_agree_correct_percentages.append(0)
        all_agree_incorrect_percentages.append(0)
        video_text_correct_percentages.append(0)
        video_text_incorrect_percentages.append(0)
        text_graph_correct_percentages.append(0)
        text_graph_incorrect_percentages.append(0)
        graph_video_correct_percentages.append(0)
        graph_video_incorrect_percentages.append(0)
        all_disagree_video_correct_percentages.append(0)
        all_disagree_text_correct_percentages.append(0)
        all_disagree_graph_correct_percentages.append(0)
        all_disagree_none_correct_percentages.append(0)

# Plot bars
# Split all_agree into correct and incorrect
plt.bar(x - width, all_agree_correct_percentages, width, label='All Agree (Correct)', color='darkgreen')
plt.bar(x - width, all_agree_incorrect_percentages, width, bottom=all_agree_correct_percentages, 
        label='All Agree (Incorrect)', color='lightgreen')

# Define colors for correct and incorrect
video_text_colors = ['#3498db', '#85c1e9']    # Bright blue and lighter blue
text_graph_colors = ['#f1c40f', '#f7dc6f']    # Bright yellow and lighter gold
graph_video_colors = ['#9b59b6', '#d7bde2']   # Bright purple and lighter lavender

# Stack the three types of two-agree with correct/incorrect split
bottom_vt = np.zeros(len(x))
plt.bar(x, video_text_correct_percentages, width, bottom=bottom_vt, 
        label='Video-Text Agree (Correct)', color=video_text_colors[0])
bottom_vt += video_text_correct_percentages
plt.bar(x, video_text_incorrect_percentages, width, bottom=bottom_vt, 
        label='Video-Text Agree (Incorrect)', color=video_text_colors[1])
bottom_vt += video_text_incorrect_percentages

plt.bar(x, text_graph_correct_percentages, width, bottom=bottom_vt, 
        label='Text-Graph Agree (Correct)', color=text_graph_colors[0])
bottom_tg = bottom_vt + text_graph_correct_percentages
plt.bar(x, text_graph_incorrect_percentages, width, bottom=bottom_tg, 
        label='Text-Graph Agree (Incorrect)', color=text_graph_colors[1])
bottom_tg += text_graph_incorrect_percentages

plt.bar(x, graph_video_correct_percentages, width, bottom=bottom_tg, 
        label='Graph-Video Agree (Correct)', color=graph_video_colors[0])
bottom_gv = bottom_tg + graph_video_correct_percentages
plt.bar(x, graph_video_incorrect_percentages, width, bottom=bottom_gv, 
        label='Graph-Video Agree (Incorrect)', color=graph_video_colors[1])

# Define colors for all disagree
all_disagree_colors = ['#8B0000', '#B22222', '#FF0000', '#FF6347']  # Dark to light red

# Stack the all disagree categories
bottom_ad = np.zeros(len(x))
plt.bar(x + width, all_disagree_video_correct_percentages, width, bottom=bottom_ad, 
        label='All Disagree (Video Correct)', color=all_disagree_colors[0])
bottom_ad += all_disagree_video_correct_percentages

plt.bar(x + width, all_disagree_text_correct_percentages, width, bottom=bottom_ad, 
        label='All Disagree (Text Correct)', color=all_disagree_colors[1])
bottom_ad += all_disagree_text_correct_percentages

plt.bar(x + width, all_disagree_graph_correct_percentages, width, bottom=bottom_ad, 
        label='All Disagree (Graph Correct)', color=all_disagree_colors[2])
bottom_ad += all_disagree_graph_correct_percentages

plt.bar(x + width, all_disagree_none_correct_percentages, width, bottom=bottom_ad, 
        label='All Disagree (None Correct)', color=all_disagree_colors[3])

# Customize plot
plt.rcParams.update({'font.size': 14})
plt.xlabel('Question Type', fontsize=16)
plt.ylabel('Percentage of Questions', fontsize=16)
plt.title('Agreement Distribution by Question Type', fontsize=18)
plt.xticks(x, labels, fontsize=14, rotation=45, ha='right')
plt.legend(fontsize=12)
plt.ylim(0, 100)
plt.grid(True, axis='y', alpha=0.3)

# Add percentage labels on bars
def add_labels(x_pos, heights, width):
    for i, height in enumerate(heights):
        if height > 1:  # Only add label if segment is at least 1%
            plt.text(x_pos[i], height, f'{int(height)}%', 
                    ha='center', va='bottom')

# Add labels for all-agree correct and incorrect
for i in range(len(x)):
    # All agree correct
    if all_agree_correct_percentages[i] > 1:
        plt.text(x[i] - width, all_agree_correct_percentages[i]/2, 
                f'{int(all_agree_correct_percentages[i])}%', 
                ha='center', va='center', fontsize=12, color='white')
    
    # All agree incorrect
    if all_agree_incorrect_percentages[i] > 1:
        mid_point = all_agree_correct_percentages[i] + all_agree_incorrect_percentages[i]/2
        plt.text(x[i] - width, mid_point, 
                f'{int(all_agree_incorrect_percentages[i])}%', 
                ha='center', va='center', fontsize=12)
    
    # Total all_agree percentage
    total_all_agree = all_agree_correct_percentages[i] + all_agree_incorrect_percentages[i]
    if total_all_agree > 1:
        plt.text(x[i] - width, total_all_agree, f'{int(total_all_agree)}%',
                ha='center', va='bottom', fontsize=12)

# Add labels for all-disagree segments
for i in range(len(x)):
    # Video correct
    if all_disagree_video_correct_percentages[i] >= 2:
        plt.text(x[i] + width, all_disagree_video_correct_percentages[i]/2, 
                f'{int(all_disagree_video_correct_percentages[i])}%', 
                ha='center', va='center', fontsize=12, color='white')
    
    # Text correct
    if all_disagree_text_correct_percentages[i] >= 2:
        mid_point = all_disagree_video_correct_percentages[i] + all_disagree_text_correct_percentages[i]/2
        plt.text(x[i] + width, mid_point, 
                f'{int(all_disagree_text_correct_percentages[i])}%', 
                ha='center', va='center', fontsize=12, color='white')
    
    # Graph correct
    if all_disagree_graph_correct_percentages[i] >= 2:
        mid_point = all_disagree_video_correct_percentages[i] + all_disagree_text_correct_percentages[i] + all_disagree_graph_correct_percentages[i]/2
        plt.text(x[i] + width, mid_point, 
                f'{int(all_disagree_graph_correct_percentages[i])}%', 
                ha='center', va='center', fontsize=12, color='white')
    
    # None correct
    if all_disagree_none_correct_percentages[i] >= 2:
        mid_point = all_disagree_video_correct_percentages[i] + all_disagree_text_correct_percentages[i] + all_disagree_graph_correct_percentages[i] + all_disagree_none_correct_percentages[i]/2
        plt.text(x[i] + width, mid_point, 
                f'{int(all_disagree_none_correct_percentages[i])}%', 
                ha='center', va='center', fontsize=12)
    
    # Total all_disagree percentage
    total_all_disagree = (all_disagree_video_correct_percentages[i] + all_disagree_text_correct_percentages[i] + 
                         all_disagree_graph_correct_percentages[i] + all_disagree_none_correct_percentages[i])
    if total_all_disagree > 1:
        plt.text(x[i] + width, total_all_disagree, f'{int(total_all_disagree)}%',
                ha='center', va='bottom', fontsize=12)

# Add labels for stacked bars (only if segment is at least 1%)
for i in range(len(x)):
    # Video-Text correct
    if video_text_correct_percentages[i] > 1:
        plt.text(x[i], video_text_correct_percentages[i]/2, 
                f'{int(video_text_correct_percentages[i])}%', 
                ha='center', va='center', fontsize=12)
    
    # Video-Text incorrect
    if video_text_incorrect_percentages[i] > 1:
        mid_point = video_text_correct_percentages[i] + video_text_incorrect_percentages[i]/2
        plt.text(x[i], mid_point, 
                f'{int(video_text_incorrect_percentages[i])}%', 
                ha='center', va='center', fontsize=12)
    
    # Text-Graph correct
    if text_graph_correct_percentages[i] > 1:
        mid_point = video_text_correct_percentages[i] + video_text_incorrect_percentages[i] + text_graph_correct_percentages[i]/2
        plt.text(x[i], mid_point, 
                f'{int(text_graph_correct_percentages[i])}%', 
                ha='center', va='center', fontsize=12)
    
    # Text-Graph incorrect
    if text_graph_incorrect_percentages[i] > 1:
        mid_point = video_text_correct_percentages[i] + video_text_incorrect_percentages[i] + text_graph_correct_percentages[i] + text_graph_incorrect_percentages[i]/2
        plt.text(x[i], mid_point, 
                f'{int(text_graph_incorrect_percentages[i])}%', 
                ha='center', va='center', fontsize=12)
    
    # Graph-Video correct
    if graph_video_correct_percentages[i] > 1:
        mid_point = video_text_correct_percentages[i] + video_text_incorrect_percentages[i] + text_graph_correct_percentages[i] + text_graph_incorrect_percentages[i] + graph_video_correct_percentages[i]/2
        plt.text(x[i], mid_point, 
                f'{int(graph_video_correct_percentages[i])}%', 
                ha='center', va='center', fontsize=12)
    
    # Graph-Video incorrect
    if graph_video_incorrect_percentages[i] > 1:
        mid_point = video_text_correct_percentages[i] + video_text_incorrect_percentages[i] + text_graph_correct_percentages[i] + text_graph_incorrect_percentages[i] + graph_video_correct_percentages[i] + graph_video_incorrect_percentages[i]/2
        plt.text(x[i], mid_point, 
                f'{int(graph_video_incorrect_percentages[i])}%', 
                ha='center', va='center', fontsize=12)
    
    # Total two-agree percentage
    total_two_agree = (video_text_correct_percentages[i] + video_text_incorrect_percentages[i] + 
                       text_graph_correct_percentages[i] + text_graph_incorrect_percentages[i] + 
                       graph_video_correct_percentages[i] + graph_video_incorrect_percentages[i])
    if total_two_agree > 1:
        plt.text(x[i], total_two_agree, f'{int(total_two_agree)}%',
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('agreement_distribution.png', dpi=300)
plt.close()




# Calculate and visualize Cohen's Kappa scores
from sklearn.metrics import cohen_kappa_score
import seaborn as sns

def calculate_kappa_matrix():
    """Calculate Cohen's Kappa agreement between modalities for all questions"""
    modalities = ['text', 'graph', 'video']
    kappa_matrix = np.zeros((3,3))
    
    # Get predictions for all questions
    predictions = {
        modality: [modality_data[modality][qid]['pred'] 
                  for qid in all_questions]
        for modality in modalities
    }
    
    # Calculate kappa for each pair
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i != j:
                kappa = cohen_kappa_score(predictions[mod1], predictions[mod2])
                kappa_matrix[i,j] = kappa
            else:
                kappa_matrix[i,j] = 1.0  # Perfect agreement with self
                
    return kappa_matrix, modalities

def calculate_kappa_by_type():
    """Calculate Cohen's Kappa agreement between modalities by question type"""
    modalities = ['text', 'graph', 'video']
    type_kappas = {}
    
    # Group questions by type
    questions_by_type = defaultdict(list)
    for qid in all_questions:
        q_type = modality_data['text'][qid]['type']
        questions_by_type[q_type].append(qid)
    
    # Calculate kappa matrices for each type
    for q_type, qids in questions_by_type.items():
        kappa_matrix = np.zeros((3,3))
        
        predictions = {
            modality: [modality_data[modality][qid]['pred'] 
                      for qid in qids]
            for modality in modalities
        }
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i != j:
                    kappa = cohen_kappa_score(predictions[mod1], predictions[mod2])
                    kappa_matrix[i,j] = kappa
                else:
                    kappa_matrix[i,j] = 1.0
                    
        type_kappas[q_type] = kappa_matrix
        
    return type_kappas, modalities
# Calculate overall kappa scores
overall_kappa, modalities = calculate_kappa_matrix()

# Calculate and plot kappa scores by question type
type_kappas, modalities = calculate_kappa_by_type()

# Set up the plot
plt.figure(figsize=(15, 8))

# Extract agreement pairs
def get_agreement_pairs(kappa_matrix):
    video_text = kappa_matrix[modalities.index('video'), modalities.index('text')]
    text_graph = kappa_matrix[modalities.index('text'), modalities.index('graph')]
    graph_video = kappa_matrix[modalities.index('graph'), modalities.index('video')]
    return [video_text, text_graph, graph_video]

# Get data for all question types and overall 
all_agreements = []
all_agreements.append(get_agreement_pairs(overall_kappa))
for q_type in sorted_q_types:
    all_agreements.append(get_agreement_pairs(type_kappas[q_type]))

# Set up bar positions
x = np.arange(len(sorted_q_types) + 1)  # +1 for overall
width = 0.25  # width of bars

# Plot bars for each pair
plt.bar(x - width, [agreements[0] for agreements in all_agreements], width, label='Video-Text')
plt.bar(x, [agreements[1] for agreements in all_agreements], width, label='Text-Graph')
plt.bar(x + width, [agreements[2] for agreements in all_agreements], width, label='Graph-Video')

# Customize plot
plt.rcParams.update({'font.size': 14})  # Increase base font size
plt.xlabel('Question Type', fontsize=16)
plt.ylabel('Cohen\'s Kappa Score', fontsize=16)
plt.title('Agreement Between Modalities by Question Type', fontsize=18)
plt.xticks(x, ['Overall'] + sorted_q_types, fontsize=14)
plt.legend(fontsize=14)
plt.ylim(0, 1)
plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('kappa_agreement_comparison.png', dpi=300)
plt.close()
