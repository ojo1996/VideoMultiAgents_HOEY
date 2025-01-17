from PIL import Image
import requests
import torch
import transformers
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import os
from pprint import pprint
import pdb

import numpy as np
from transformers import AutoModel, AutoConfig
from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer
import torchvision.transforms as T



def extract_features(image_dir):
    ###### ORIGINAL MODEL ######
    model_name_or_path = os.getenv("EVA-CLIP-8B") #"BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-8B
    image_size = 224
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True).to(device).eval()
    resume = True
    
    image_dir_path = Path(image_dir)
    image_paths = list(image_dir_path.iterdir())
    image_paths = [path for path in image_paths if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    image_paths.sort(key=lambda x: int(x.stem.split('_')[-1]))
    img_feature_list = []
    for image_path in image_paths:
        image = Image.open(str(image_path))

        input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(input_pixels)
            img_feature_list.append(image_features)
    img_feature_tensor = torch.stack(img_feature_list)
    img_feats = img_feature_tensor.squeeze(1)
    return img_feats

if __name__ == '__main__':
    extract_features(image_dir)