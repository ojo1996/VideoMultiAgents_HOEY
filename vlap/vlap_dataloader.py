import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json

class NextQADataset(Dataset):
    def __init__(self, csv_file, features_dir, json_file, max_length=128):
        self.csv_data = pd.read_csv(csv_file)
        self.features_dir = features_dir
        self.max_length = max_length
        self.json_data = json.load(open(json_file))  # JSONデータの読み込み

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]

        # JSONを使用してビデオIDを取得
        video_key = str(row['video'])  # CSVのvideoフィールド
        video_id = self.json_data.get(video_key, None)
        if not video_id:
            raise ValueError(f"No mapping found for video ID {video_key}")
        
        video_id_last = video_id.split('/')[-1]
        
        # ビデオ特徴の読み込み
        feature_filepath = os.path.join(self.features_dir, f'{video_id}', f'{video_id_last}_features.pt')
        if not os.path.exists(feature_filepath):
            raise FileNotFoundError(f"Feature file not found: {feature_filepath}")

        video_features = torch.load(feature_filepath)

        # 質問テキストの取得（トークナイズせずにそのまま）
        question = row['question']
        
        # 正解の回答（番号）を取得
        answer = int(row['answer'])
        
        # 選択肢をそのまま取得してリストとして保持
        options = [row[f'a{i}'] for i in range(5)]
        
        # print(f"Processing video: {video_id}, Question: {question}, Options: {options}")

        
        return {
            'video_features': video_features,  # テンソル型のビデオ特徴
            'question': question,  # そのままの質問テキスト
            'answer': answer,  # 正解のインデックス
            'options': options  # そのままの選択肢テキスト
        }

def get_nextqa_loader(csv_file, features_dir, json_file, batch_size=8, shuffle=True, pin_memory=True):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = NextQADataset(csv_file, features_dir, json_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    
    return dataloader

# 使用例
# csv_file = '/path/to/nextqa/train.csv'
# features_dir = '/path/to/nextqa/NExTVideoFeatures'
# json_file = '/path/to/nextqa/map_vid_vidorID.json'
# dataloader = get_nextqa_loader(csv_file, features_dir, json_file, batch_size=8)