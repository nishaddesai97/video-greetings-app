from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
import cv2
import numpy as np
import os

def create_text_frame(text, size, font_scale=1, thickness=2):
    """Create a frame with text overlay using OpenCV"""
    frame = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate position to center text
    x = (size[0] - text_width) // 2
    y = (size[1] + text_height) // 2
    
    # Add background
    cv2.rectangle(frame, 
                 (x - 10, y - text_height - 10),
                 (x + text_width + 10, y + 10),
                 (0, 0, 0, 128), 
                 -1)
    
    # Add text
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255, 255), thickness)

def compile_greeting_video(campaign, app_config):
    """
    Compile multiple greeting videos into one with smooth transitions
    """
    app_dir = os.path.abspath(os.path.dirname(__file__))
    temp_dir = os.path.join(app_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    final_clips = []
    target_size = (720, 1280)  # Standard vertical video size
    
    # Process each video
    for video in campaign['videos']:
        input_path = os.path.join(app_dir, video['path'])
        if not os.path.exists(input_path):
            print(f"Warning: Video file not found: {input_path}")
            continue
        
        try:
            # Load and resize video
            clip = VideoFileClip(input_path)
            
            # Resize and crop to target size
            w, h = clip.size
            if w/h > target_size[0]/target_size[1]:  # Too wide
                new_w = h * target_size[0]/target_size[1]
                crop_w = (w - new_w)/2
                clip = clip.crop(x1=crop_w, x2=w-crop_w)
            elif w/h < target_size[0]/target_size[1]:  # Too tall
                new_h = w * target_size[1]/target_size[0]
                crop_h = (h - new_h)/2
                clip = clip.crop(y1=crop_h, y2=h-crop_h)
            
            clip = clip.resize(target_size)
            
            # Create a function to process each frame
            def add_text_to_frame(frame):
                height, width = frame.shape[:2]
                text_frame = create_text_frame(f"From: {video['name']}", (width, height))
                # Combine frame with text overlay
                alpha = text_frame[:,:,3:] / 255.0
                frame = frame * (1 - alpha) + text_frame[:,:,:3] * alpha
                return frame.astype('uint8')
            
            # Apply text overlay to video
            final_clip = clip.fl(add_text_to_frame)
            final_clips.append(final_clip)
            
        except Exception as e:
            print(f"Error processing video {input_path}: {str(e)}")
            continue
    
    if not final_clips:
        raise Exception("No videos could be processed successfully")
    
    # Create title clip
    ai_message = campaign.get('ai_message', 'Thank you for your message!')
    ai_emojis = campaign.get('ai_emojis', '🎉 ✨ 💝')
    
    def create_title_frame(size=(720, 1280)):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Split message into lines
        lines = [ai_message, ai_emojis]
        y_position = size[1] // 3  # Start text at 1/3 from top
        
        for line in lines:
            # Get text size
            font_scale = 1.5
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            
            # Calculate position to center text
            x = (size[0] - text_width) // 2
            
            # Add background
            cv2.rectangle(frame, 
                         (x - 20, y_position - text_height - 20),
                         (x + text_width + 20, y_position + 20),
                         (0, 0, 0), 
                         -1)
            
            # Add text
            cv2.putText(frame, line, (x, y_position), font, font_scale, (255, 255, 255), thickness)
            y_position += text_height + 40  # Space between lines
        
        return frame
    
    # Create title frames for 3 seconds at 30fps
    title_frames = [create_title_frame() for _ in range(90)]  # 3 seconds * 30 fps
    
    # Create title clip
    from moviepy.video.VideoClip import ImageClip
    title_clips = [ImageClip(frame, duration=1/30) for frame in title_frames]
    title_txt = concatenate_videoclips(title_clips)
    
    # Add transitions
    transition_duration = 0.5
    clips_with_transitions = []
    
    for i, clip in enumerate(final_clips):
        if i == 0:  # First clip
            clips_with_transitions.append(clip.crossfadeout(transition_duration))
        elif i == len(final_clips) - 1:  # Last clip
            clips_with_transitions.append(clip.crossfadein(transition_duration))
        else:  # Middle clips
            clips_with_transitions.append(
                clip.crossfadein(transition_duration).crossfadeout(transition_duration)
            )
    
    # Concatenate all clips
    final_video = concatenate_videoclips(
        [title_txt] + clips_with_transitions,
        method="compose",
        padding=-transition_duration
    )
    
    # Write final video
    output_path = os.path.join(app_dir, app_config['COMPILED_FOLDER'], f"{campaign['id']}_final.mp4")
    
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=30,
        preset='medium',
        ffmpeg_params=[
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'main',
            '-level', '3.1',
            '-bf', '2',
            '-crf', '23',
            '-maxrate', '4M',
            '-bufsize', '8M',
            '-movflags', '+faststart'
        ],
        threads=4
    )
    
    # Clean up
    for clip in final_clips:
        clip.close()
    
    return output_path
