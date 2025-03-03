import cv2
import numpy as np
import os
import subprocess as sp
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def extract_frames_to_collage(video_path, output_path, start_time="00:00", end_time=None, num_frames=4, grid_size=(2, 2), output_size=384):
    """
    Extract frames from a video between specified start and end times and create a collage.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the output collage
        start_time: Start time in format "mm:ss"
        end_time: End time in format "mm:ss", if None, uses the end of the video
        num_frames: Number of frames to extract uniformly between start and end times
        grid_size: Tuple of (rows, cols) for the collage grid
        output_size: Size of the output image (width and height in pixels)
    """
    # Get video properties using FFprobe
    cmd_probe = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,duration',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    
    try:
        probe_output = sp.check_output(cmd_probe).decode().strip()
        width, height, avg_frame_rate, duration = probe_output.split(',')
        
        # Parse frame rate (handle fractions like 30000/1001)
        if '/' in avg_frame_rate:
            num, den = avg_frame_rate.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(avg_frame_rate)
            
        total_duration = float(duration)
        total_frames = int(total_duration * fps)
    except Exception as e:
        print(f"Error getting video properties: {e}")
        # Fallback to OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    
    # Parse start time
    start_min, start_sec = map(int, start_time.split(':'))
    start_time_seconds = start_min * 60 + start_sec
    start_frame = int(start_time_seconds * fps)
    
    # Parse end time or use video end
    if end_time is None:
        end_frame = total_frames - 1
    else:
        end_min, end_sec = map(int, end_time.split(':'))
        end_time_seconds = end_min * 60 + end_sec
        end_frame = int(end_time_seconds * fps)
    
    # Ensure frames are within valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))
    
    # Calculate frame indices to extract (uniform sampling)
    if num_frames == 1:
        frame_indices = [start_frame]
    else:
        frame_indices = [start_frame + i * (end_frame - start_frame) // (num_frames - 1) for i in range(num_frames)]
    
    # Ensure we don't exceed the number of frames needed for the grid
    max_frames = grid_size[0] * grid_size[1]
    frame_indices = frame_indices[:max_frames]
    
    # Extract frames using FFmpeg
    frames = []
    timestamps = []
    
    for frame_idx in frame_indices:
        timestamp_sec = frame_idx / fps
        seek_time = timestamp_sec
        
        # Format timestamp as mm:ss
        minutes = int(timestamp_sec // 60)
        seconds = int(timestamp_sec % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        # Use FFmpeg to extract a single frame
        cmd = [
            'ffmpeg',
            '-ss', str(seek_time),
            '-i', str(video_path),
            '-vframes', '1',
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-'
        ]
        
        try:
            pipe = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**8)
            raw_frame = pipe.stdout.read()
            pipe.stdout.close()
            pipe.wait()
            
            if raw_frame:
                # Ensure width and height are even numbers for FFmpeg compatibility
                width_even = int(width) - (int(width) % 2)
                height_even = int(height) - (int(height) % 2)
                
                frame = np.frombuffer(raw_frame, dtype='uint8').reshape((height_even, width_even, 3))
                pil_image = Image.fromarray(frame)
                frames.append(pil_image)
                timestamps.append(timestamp)
        except Exception as e:
            # print(f"Error extracting frame at {timestamp}: {e}")
            # Alternative: Extract frames using OpenCV
            # print("Attempting to extract frames using OpenCV...")
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return
            
            for frame_idx in frame_indices:
                # Set the position of the next frame to be read
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Format timestamp
                    timestamp_sec = frame_idx / fps
                    minutes = int(timestamp_sec // 60)
                    seconds = int(timestamp_sec % 60)
                    timestamp = f"{minutes:02d}:{seconds:02d}"
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                    timestamps.append(timestamp)
                else:
                    print(f"Error: Could not read frame at index {frame_idx}")
            
            cap.release()
    
    # Create a collage
    if frames:
        # Calculate frame dimensions for the collage
        frame_width = output_size // grid_size[1]
        frame_height = output_size // grid_size[0]
        
        # Resize frames to fit the grid
        pil_frames = [frame.resize((frame_width, frame_height)) for frame in frames]
        
        # Create a blank canvas for the collage
        collage_width = frame_width * grid_size[1]
        collage_height = frame_height * grid_size[0]
        collage = Image.new('RGB', (collage_width, collage_height))
        
        # Try to load a font, fall back to default if not available
        font_size = max(10, output_size // 40)  # Scale font size based on output size
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
        # Place frames in the collage
        for i, (frame, timestamp) in enumerate(zip(pil_frames, timestamps)):
            row = i // grid_size[1]
            col = i % grid_size[1]
            
            x = col * frame_width
            y = row * frame_height
            
            collage.paste(frame, (x, y))
            
            # Add timestamp label
            draw = ImageDraw.Draw(collage)
            draw.text((x + 5, y + 5), timestamp, fill=(255, 0, 0), font=font)
        
        # Save the collage
        collage.save(output_path)
        print(f"Collage saved to {output_path}")
    else:
        print("No frames were extracted")


if __name__ == "__main__":
    # Path to the video file
    video_path = "/simurgh/u/akhatua/VideoMultiAgents/data/nextqa/NExTVideo/1009/7416295940.mp4"

    # Path to save the output collage
    output_path = "demo.png"

    # Extract frames and create a 4x4 collage with default output size of 384
    extract_frames_to_collage(video_path, output_path, start_time="01:08", end_time="01:16", num_frames=16, grid_size=(4, 4), output_size=768)
