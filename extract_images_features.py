import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import pandas as pd
import cv2
import os
import sys

def sample_frames_from_video(video_path, output_folder, num_frames=32):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Error: Video file {video_path} does not have any frames")
        return
    
    step = total_frames // num_frames
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    saved_frames = 0
    
    while saved_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frames + 1}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
        frame_count += 1
    
    cap.release()
    print(f"{saved_frames} frames have been saved to {output_folder}")

# DINOv2モデルのロード
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
model.to('cuda')
model.eval()

# 画像の前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CSVファイルの読み込み
csv_data = pd.read_csv('/root/project_ws/VideoMultiAgents/dataset/nextqa/nextqa/val.csv')

# JSONファイルの読み込み
json_data = json.load(open('/root/project_ws/VideoMultiAgents/dataset/nextqa/nextqa/map_vid_vidorID.json'))

# CSVファイルの各行をループ処理
for index, row in csv_data.iterrows():
    video_id = json_data.get(str(row['video']), None)
    if not video_id:
        print(f"Error: No mapping found for video ID {row['video']}")
        continue
    
    videofile_path = f'/root/project_ws/VideoMultiAgents/dataset/nextqa/NExTVideo/{video_id}.mp4'
    if not os.path.exists(videofile_path):
        print(f"Error: Cannot open video file {videofile_path}")
        continue
    
    output_folder = f'/root/project_ws/VideoMultiAgents/dataset/nextqa/NExTVideoFrames/{video_id}'
    
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) == 32:
        # print(f"Skip: Folder {output_folder} already exists and contains 32 frames")
        pass
    else:
        sample_frames_from_video(videofile_path, output_folder, num_frames=32)

    # create folder for features file if not exists
    output_folder_for_binary = f'/root/project_ws/VideoMultiAgents/dataset/nextqa/NExTVideoFeatures/{video_id}'
    if not os.path.exists(output_folder_for_binary):
        os.makedirs(output_folder_for_binary)
    # Check if features file already exists
    video_name = row['video']
    feature_filename = f'{video_name}_features.pt'
    feature_filepath = os.path.join(output_folder_for_binary, feature_filename)
    if os.path.exists(feature_filepath):
        print(f"Skip: Features file {feature_filepath} already exists")
        continue

    # Extract features from frames
    frames = []
    for i in range(1, 33):
        frame_path = os.path.join(output_folder, f"frame_{i}.png")
        image = Image.open(frame_path)
        input_tensor = transform(image)
        frames.append(input_tensor)
    frames_batch = torch.stack(frames).to('cuda')

    with torch.no_grad():
        features_batch = model(frames_batch)
    
    print (features_batch.shape)
    torch.save(features_batch.cpu(), feature_filepath)
    print(f"Features have been saved to {feature_filepath}")
