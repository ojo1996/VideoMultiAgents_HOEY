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
    # Initialize default values
    width, height, fps, total_frames = None, None, None, None
    
    # Get video properties using FFprobe first
    cmd_probe = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,duration',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    
    try:
        probe_output = sp.check_output(cmd_probe).decode().strip()
        parts = probe_output.split(',')
        
        # Check if we have all the expected parts
        if len(parts) == 4:
            width_str, height_str, avg_frame_rate, duration_str = parts
            
            # Handle potential 'N/A' values
            if width_str != 'N/A' and height_str != 'N/A':
                width = int(float(width_str))
                height = int(float(height_str))
            
            # Parse frame rate (handle fractions like 30000/1001)
            if avg_frame_rate != 'N/A':
                if '/' in avg_frame_rate:
                    num, den = avg_frame_rate.split('/')
                    fps = float(num) / float(den)
                else:
                    fps = float(avg_frame_rate)
            
            # Parse duration
            if duration_str != 'N/A':
                total_duration = float(duration_str)
                if fps is not None:
                    total_frames = int(total_duration * fps)
    except Exception as e:
        print(f"Error getting video properties with FFprobe: {e}")
    
    # Ensure we have valid values
    if width <= 0 or height <= 0 or fps <= 0 or total_frames <= 0:
        print(f"Invalid video properties: width={width}, height={height}, fps={fps}, total_frames={total_frames}")
        return False
    
    print(f"Video properties: width={width}, height={height}, fps={fps}, total_frames={total_frames}")
    
    # Parse start time
    try:
        start_min, start_sec = map(int, start_time.split(':'))
        start_time_seconds = start_min * 60 + start_sec
        start_frame = int(start_time_seconds * fps)
    except Exception as e:
        print(f"Error parsing start time {start_time}: {e}")
        start_frame = 0
    
    # Parse end time or use video end
    if end_time is None:
        end_frame = total_frames - 1
    else:
        try:
            end_min, end_sec = map(int, end_time.split(':'))
            end_time_seconds = end_min * 60 + end_sec
            end_frame = int(end_time_seconds * fps)
        except Exception as e:
            print(f"Error parsing end time {end_time}: {e}")
            end_frame = total_frames - 1
    
    # Ensure frames are within valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))
    
    # Calculate available frames
    available_frames = end_frame - start_frame + 1
    
    # Adjust num_frames if it's larger than available frames
    if num_frames > available_frames:
        print(f"Requested {num_frames} frames but only {available_frames} frames are available. Adjusting.")
        num_frames = available_frames
    
    # Calculate frame indices to extract (uniform sampling)
    if num_frames == 1:
        frame_indices = [start_frame]
    elif num_frames >= available_frames:
        # Take every frame from start_frame to end_frame
        frame_indices = list(range(start_frame, end_frame + 1))
    else:
        frame_indices = [start_frame + i * (end_frame - start_frame) // (num_frames - 1) for i in range(num_frames)]
    
    # Ensure we don't exceed the number of frames needed for the grid
    max_frames = grid_size[0] * grid_size[1]
    frame_indices = frame_indices[:max_frames]
    
    # Extract frames using OpenCV (more reliable than pipe-based FFmpeg approach)
    frames = []
    timestamps = []
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
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
                print(f"Warning: Could not read frame at index {frame_idx}")
        
        cap.release()
    except Exception as e:
        print(f"Error extracting frames with OpenCV: {e}")
        
    # If OpenCV failed, try using FFmpeg as fallback (optional)
    if not frames:
        print("Attempting to extract frames using FFmpeg...")
        for frame_idx in frame_indices:
            timestamp_sec = frame_idx / fps
            seek_time = timestamp_sec
            
            # Format timestamp as mm:ss
            minutes = int(timestamp_sec // 60)
            seconds = int(timestamp_sec % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            
            # Use FFmpeg to extract a single frame to a temporary file
            temp_output = f"temp_frame_{frame_idx}.jpg"
            cmd = [
                'ffmpeg',
                '-ss', str(seek_time),
                '-i', str(video_path),
                '-vframes', '1',
                '-q:v', '2',
                temp_output
            ]
            
            try:
                sp.call(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                
                if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                    pil_image = Image.open(temp_output)
                    frames.append(pil_image)
                    timestamps.append(timestamp)
                    os.remove(temp_output)
            except Exception as e:
                print(f"Error extracting frame at {timestamp} with FFmpeg: {e}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)
    
    # Create a collage
    if frames:
        # Ensure we have at least one frame
        if len(frames) < grid_size[0] * grid_size[1]:
            print(f"Warning: Only {len(frames)} frames were extracted, but the grid requires {grid_size[0] * grid_size[1]}")
        
        # Calculate frame dimensions for the collage
        frame_width = output_size // grid_size[1]
        frame_height = output_size // grid_size[0]
        
        # Resize frames to fit the grid
        pil_frames = []
        for frame in frames:
            try:
                resized_frame = frame.resize((frame_width, frame_height))
                pil_frames.append(resized_frame)
            except Exception as e:
                print(f"Error resizing frame: {e}")
        
        # Create a blank canvas for the collage
        collage_width = frame_width * grid_size[1]
        collage_height = frame_height * grid_size[0]
        collage = Image.new('RGB', (collage_width, collage_height))
        
        # Try to load a font, fall back to default if not available
        font_size = max(10, output_size // 40)  # Scale font size based on output size
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                # Try some other common fonts
                fonts_to_try = ["DejaVuSans.ttf", "FreeSans.ttf", "LiberationSans-Regular.ttf"]
                font = None
                for font_name in fonts_to_try:
                    try:
                        font = ImageFont.truetype(font_name, font_size)
                        break
                    except:
                        continue
                if font is None:
                    font = ImageFont.load_default()
            except:
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
        try:
            collage.save(output_path)
            print(f"Collage saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving collage: {e}")
            return False
    else:
        print("No frames were extracted")
    
    return False


if __name__ == "__main__":
    # Path to the video file
    video_path = "/simurgh/u/akhatua/VideoMultiAgents/data/nextqa/NExTVideo/1009/7416295940.mp4"

    # Path to save the output collage
    output_path = "demo.png"

    # Extract frames and create a 4x4 collage with default output size of 384
    extract_frames_to_collage(video_path, output_path, start_time="01:08", end_time="01:16", num_frames=16, grid_size=(4, 4), output_size=768)