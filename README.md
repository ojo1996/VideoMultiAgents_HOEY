# VideoMultiAgent

VideoMultiAgent for Long-Term video understanding.

## 1 Prerequisites

- Docker installed

## 2 Preparation

### 2.1 Download the Datasets

#### 2.1.1 EgoSchemaVQA dataset

##### Please refer to the following official repository to download the EgoSchemaVQA dataset.
https://github.com/egoschema/EgoSchema

#### You can download the Question file of EgoSchemaVQA dataset from the following link:
This link is from LLoVi's github.
https://drive.google.com/file/d/13M10CB5ePPVlycn754_ff3CwnpPtDfJA/view?usp=drive_link

#### 2.1.2 NextQA dataset

Raw videos for train/val/test are available at: (https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view).

you may need map_vid_vidorID.json to find the videos, avaialble at: https://drive.google.com/file/d/1NFAOQYZ-D0LOpcny8fm0fSzLvN0tNR2A/view?usp=sharing

### 2.2 LLoVi Caption Data

Our model uses the LLoVi caption data. You can download the LLoVi caption data from the following link.
https://github.com/CeeZh/LLoVi

Then use the extract_images_features.py to convert Video files for NextQA dataset into frames and csv2json.py to convert val.csv to nextqa.json

## 3 Setup & Usage

### 3.1 set the environment variables

Our model uses the OpenAI. So, Please set the access infomation into the Dockerfile.

### 3.2 set the variables

Set the appropriate file paths inside main.py

## 4 Build & Run

### 4.1 Build the container

`docker compose up`

### 4.2 Run the container

`docker exec -it docker-video_multi_agents_env-1 /bin/bash`

### 4.3 Run the python script from inside the cntainer

`python3 main.py --dataset=DATASET`