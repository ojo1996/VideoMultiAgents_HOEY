import os
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from vlap_dataloader import NextQADataset, get_nextqa_loader  # `NextQADataset`と`get_nextqa_loader`が定義されているファイルをインポート

# ファイルパスの設定（適宜変更してください）
csv_file = '/root/project_ws/VideoMultiAgents/dataset/nextqa/nextqa/train.csv'
features_dir = '/root/project_ws/VideoMultiAgents/dataset/nextqa/NExTVideoFeatures'
json_file = '/root/project_ws/VideoMultiAgents/dataset/nextqa/nextqa/map_vid_vidorID.json'

# データローダーの初期化
dataloader = get_nextqa_loader(csv_file, features_dir, json_file, batch_size=8)

# 一バッチだけ取り出して形状を表示
batch = next(iter(dataloader))

# ビデオ特徴の形状
print("Video Features Shape:", batch['video_features'].shape)

# 質問テキストの形状
print("Question Input IDs Shape:", batch['question']['input_ids'].shape)
print("Question Attention Mask Shape:", batch['question']['attention_mask'].shape)

# 正解の回答
print("Answer Shape:", torch.tensor(batch['answer']).shape)
print (batch['answer'])

# 各選択肢のエンコーディングの形状
for i, option_encoding in enumerate(batch['options']):
    print(f"Option {i} Input IDs Shape:", option_encoding['input_ids'].shape)
    print(f"Option {i} Attention Mask Shape:", option_encoding['attention_mask'].shape)