import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import gc
import os

def optimize_video_size(input_path, output_path, target_size_mb=10):
    """Optimize video size while maintaining quality"""
    clip = VideoFileClip(input_path)
    
    # Calculate current size in MB
    current_size = os.path.getsize(input_path) / (1024 * 1024)
    
    if current_size <= target_size_mb:
        clip.close()
        return input_path
    
    # Calculate compression ratio
    compression_ratio = target_size_mb / current_size
    
    # Adjust resolution and bitrate
    new_width = int(clip.w * np.sqrt(compression_ratio))
    new_width = new_width - (new_width % 2)  # Ensure even dimensions
    new_height = int(clip.h * np.sqrt(compression_ratio))
    new_height = new_height - (new_height % 2)  # Ensure even dimensions
    
    # Resize and export with reduced bitrate
    resized_clip = clip.resize((new_width, new_height))
    resized_clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        bitrate=f'{int(1000 * compression_ratio)}k',
        preset='faster'
    )
    
    # Clean up
    clip.close()
    resized_clip.close()
    gc.collect()
    
    return output_path

def process_video_in_chunks(input_path, output_path, chunk_size_frames=30):
    """Process video in smaller chunks to reduce memory usage"""
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(0, total_frames, chunk_size_frames):
        # Process chunk_size_frames at a time
        for _ in range(min(chunk_size_frames, total_frames - frame_idx)):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
                
        # Force garbage collection after each chunk
        gc.collect()
    
    # Clean up
    cap.release()
    out.release()
    gc.collect()
    
    return output_path
