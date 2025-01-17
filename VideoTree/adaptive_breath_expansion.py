import os
from pathlib import Path
from .util import *
from eval import *
from .dataset import get_dataset
from .prompts import PromptFactory
from .model import get_model
from tqdm import tqdm
from pprint import pprint
from .kmeans_pytorch.kmeans_pytorch import kmeans
import torch


def find_closest_points_per_cluster(x, cluster_ids, cluster_centers):
    # Dictionary to store the indices of the closest points for each cluster
    closest_points_idx_per_cluster = {cluster_id: [] for cluster_id in range(len(cluster_centers))}
    
    # Iterate over each cluster
    for cluster_id in range(len(cluster_centers)):
        # Filter points belonging to the current cluster
        indices_in_cluster = torch.where(cluster_ids == cluster_id)[0]
        points_in_cluster = x[indices_in_cluster]
        
        # Calculate distances from points in the cluster to the cluster center
        distances = torch.norm(points_in_cluster - cluster_centers[cluster_id], dim=1)

        if distances.numel() > 0:    
            
            # Find the index (within the cluster) of the point closest to the cluster center
            closest_idx_in_cluster = torch.argmin(distances).item()
            
            # Map back to the original index in x
            closest_global_idx = indices_in_cluster[closest_idx_in_cluster].item()
            
            # Store the global index
            closest_points_idx_per_cluster[cluster_id].append(closest_global_idx)

    return closest_points_idx_per_cluster

def adaptive_breath_expansion(img_feats,video_id):

    args = parse_args()
    
    processed = {}

    # # get input
    quids_to_include = video_id
    dataset = get_dataset(args, quids_to_include=quids_to_include, num_examples_to_run=args.num_examples_to_run)

    # configure prompt
    prompter = PromptFactory().get(args.prompt_type)

    # get model
    model = get_model(args)
    model.set_post_process_fn(prompter.post_process_fn)
    
    # save width expansion results
    all_width_res = []

    # answer

    for i, item in enumerate(dataset):
        ukey_1 = item['quid'] if 'quid' in item else item['uid']

        #init the cluster parameters
        tree_node = [0]
        max_cluster_num = args.max_cluster_num
        cluster_num = args.init_cluster_num
        iter_threshold = args.iter_threshold
        adaptive_rate = args.default_adpative_rate


        clip_length = int(1/args.fps) if args.fps < 1 else 1/args.fps
        few_shot_examples = build_fewshot_examples(args.fewshot_example_path, args.data_path)

        # load frame features
        frame_feats = img_feats
        ### adaptive width expansion
        while(True):
            # width expansion
            cluster_ids_x, cluster_centers = kmeans(X=frame_feats, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:0'))
            # send cluster_ids_x to GPU 
            cluster_ids_x = cluster_ids_x.to('cuda')
            cluster_centers = cluster_centers.to('cuda')
            closest_points_idx_per_cluster = find_closest_points_per_cluster(frame_feats, cluster_ids_x, cluster_centers)
            if closest_points_idx_per_cluster is None:
                # print("closest_points_idx_per_cluster is None")
                continue
            tree_node = sorted([value for sublist in closest_points_idx_per_cluster.values() for value in sublist])

            cluster_ids_x = cluster_ids_x.tolist()
            # relevance scoring
            model.set_post_process_fn(prompter.post_process_fn)
            prompt = prompter.fill(**item, fps=args.fps, clip_length=clip_length, num_words=args.num_words_in_sum, examplars=few_shot_examples, loc_pred = tree_node)
            pred, info = model.forward(prompter.head, prompt)
            ukey_name = 'quid' if 'quid' in item else 'uid'

            # the output is the predicted frame relevance
            frame_relevance = pred
            high_relevance_frame_num = frame_relevance.count(3)

            if high_relevance_frame_num < iter_threshold:
                if cluster_num < max_cluster_num:
                    cluster_num = cluster_num * adaptive_rate
                else:
                    break
            else:
                break
        all_width_res.append({"name": ukey_1, "tree_node": tree_node, "cluster_ids_x": cluster_ids_x})


        ukey = item[ukey_name]

        processed[ukey] = item

        processed[ukey]['prompt'] = prompt
        processed[ukey]['prompt_template'] = prompter.get_template_str()
        processed[ukey]['response'] = info['response']
        processed[ukey]['pred'] = pred
        relevance_score = {"data": processed}

    return relevance_score, all_width_res


if __name__ == '__main__':
    adaptive_breath_expansion(img_feats,video_id)
    
    