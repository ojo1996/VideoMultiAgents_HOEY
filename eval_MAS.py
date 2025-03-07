from pathlib import Path
from pprint import pprint
from collections import Counter, defaultdict
import argparse
import pdb
import json

# category file for egoschema
egoschema_cats_path = "/path/to/egoschema_categories.json"

# file containes the evaluation data (questions and responses)
anno_file_path = "/path/to/anno_file.json"


def load_json(file_path):
    """Loads a JSON file and returns its content."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def eval_egoschema(anno_file_path):
    data = load_json(anno_file_path)
    num_valids = 0
    num_corrects = 0
    for uid, el in data.items():
        if el['pred'] == -1:
            continue
        num_valids += 1
        if el['truth'] == el['pred']:
            num_corrects += 1 
    stat = {
        'num_total': len(data),
        'num_valids': num_valids,
        'num_corrects': num_corrects,
        'acc': num_corrects / len(data),
    }
    pprint(stat)
    stat['data'] = data
    return stat

def eval_egoschema_cats(anno_file_path, egoschema_cats_path):
    data = load_json(anno_file_path)
    if 'data' in data:
        data = data['data']
    cats = load_json(egoschema_cats_path)
    cats = {el[1]: el[-1] for el in cats}  # uid --> [cat0, cat1, ...]

    def eval(preds):
        num_corrects = defaultdict(int)  # q_type --> int
        num_total = defaultdict(int)  # q_type --> int
        for uid, info in preds.items():
            q_type_list = info['type']
            pred = info['pred']
            truth = info['truth']
            for q_type in q_type_list:
                if pred == -1:
                    continue
                else:
                    num_corrects[q_type] += (pred==truth)
                num_total[q_type] += 1
        accs = {k: num_corrects[k] / num_total[k] for k in num_corrects}
        acc_all = sum(list(num_corrects.values())) / sum(list(num_total.values()))
        return accs, acc_all

    for k, v in cats.items():
        for el in v:
            if el not in [1, 2, 3, 4, 5]:
                print('question category not found: ', k)

    # category stat
    id_to_name = {
        1: 'Purpose/Goal Identification',
        2: 'Character Interaction',
        3: 'Tools and Materials Usage',
        4: 'Key Action/Moment Detection',
        5: 'Action Sequence Analysis'
    }
    arr = sum(list(cats.values()), [])
    stat = Counter(arr).most_common()
    print('Category Statistics:')
    for q_type, count in stat:
        print(f"{id_to_name[q_type]}: {count / len(cats) * 100:.1f}")
    print()

    # eval
    preds = {uid: {'pred': uid_info['pred'], 'truth': uid_info['truth'], 'type': cats[uid]} for uid, uid_info in data.items() if uid in cats}
    accs, acc_all = eval(preds)
    accs = sorted(list(accs.items()))

    print('Evaluation:')
    for k, v in accs:
        print(f"{id_to_name[k]}: {v*100:.1f}")
    print()
    print(f"all: {acc_all*100:.1f}")

def eval_nextqa(anno_file_path):
    '''
    This function was adapted from https://github.com/doc-doc/NExT-QA/blob/main/eval_mc.py
    '''
    map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}
    data = load_json(anno_file_path)
    group = {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}
    for qns_id, el in data.items():
        # qns_id = str(row['video']) + '_' + str(row['qid'])
        if 'pred' in el.keys() and el['pred'] >= 0:
            qtype = str(el['type'])
            #(combine temporal qns of previous and next as 'TN')
            if qtype == 'TP': qtype = 'TN'
            group[qtype].append(qns_id)

    group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    overall_acc = {'C':0, 'T':0, 'D':0}
    overall_cnt = {'C':0, 'T':0, 'D':0}
    all_acc = 0
    all_cnt = 0
    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids:

            cnt += 1
            answer = data[qid]['truth']
            pred = data[qid]['pred']

            if answer == pred: 
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt

    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    stat = {}
    for qtype in group_acc:
        print(map_name[qtype], end='\t')
    print('')
    for qtype, acc in group_acc.items():
        if group_cnt[qtype] == 0:
            stat[qtype] = 0
            print('{:.2f}'.format(0), end ='\t')
        else:
            stat[qtype] = acc*100.0/group_cnt[qtype]
            print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end ='\t')
    stat['Acc'] = all_acc*100.0/all_cnt
    print('')
    print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))
    stat['data'] = pred
    return stat

def eval_intentqa(anno_file_path):
    '''
    This function was adapted from https://github.com/JoseponLee/IntentQA/blob/main/eval_intentqa.py 
    '''
    
    # Mapping of question type codes to their descriptive names
    map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}
    
    # Load the anno_file
    anno_file = load_json(anno_file_path)

    # Initialize groups for each question type
    group = {key: [] for key in ['CW', 'CH', 'TN', 'TC', 'DC', 'DL', 'DO']}

    for q, data in anno_file.items():
        qns_id = data["q_uid"]
        qtype = data["type"]

        # Combine temporal questions of previous and next as 'TN'
        if qtype == 'TP':
            qtype = 'TN'

        # Append the question ID to the appropriate group
        if qtype in group:
            group[qtype].append(qns_id)

    # Initialize accuracy and count dictionaries
    group_acc = {key: 0 for key in ['CW', 'CH', 'TN', 'TC', 'DC', 'DL', 'DO']}
    group_cnt = {key: 0 for key in ['CW', 'CH', 'TN', 'TC', 'DC', 'DL', 'DO']}
    overall_acc = {'C': 0, 'T': 0, 'D': 0}
    overall_cnt = {'C': 0, 'T': 0, 'D': 0}
    all_acc = 0
    all_cnt = 0

    # Calculate accuracy for each group
    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids:
            cnt += 1
            answer = anno_file[qid]['truth']
            pred = anno_file[qid]['pred']
            if answer == pred:
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt

    # Update overall accuracy and counts
    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    # Filter out qtypes with zero counts
    filtered_qtypes = [qtype for qtype in group_acc if group_cnt[qtype] > 0]

    # Print the header for valid qtypes
    for qtype in filtered_qtypes:
        print(map_name[qtype], end='\t')
    print('')

    # Print the accuracy for valid qtypes
    for qtype in filtered_qtypes:
        acc = group_acc[qtype]
        cnt = group_cnt[qtype]
        accuracy = (acc * 100.0 / cnt) if cnt > 0 else 0.00
        print('{:.2f}'.format(accuracy), end='\t')
    print('')

    # Print overall accuracy
    overall_accuracy = (all_acc * 100.0 / all_cnt) if all_cnt > 0 else 0.00
    print('Acc: {:.2f}'.format(overall_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function",
        required=True,
        type=str,
    )
    args, unknown = parser.parse_known_args()
    function_arg_names = unknown[0::2]
    function_arg_values = unknown[1::2]
    function_args = {function_arg_names[i]: function_arg_values[i] for i in range(len(function_arg_names))}

    print()

    if args.function in globals():
        function = globals()[args.function]
 
        if function == eval_egoschema_cats:
            function(anno_file_path, egoschema_cats_path)  
        else:
            function(anno_file_path)

    else:
        print(f"Function {args.function} not found.")