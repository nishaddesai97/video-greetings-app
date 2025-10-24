import os
import cv2
from moviepy.editor import (VideoFileClip, concatenate_videoclips, AudioFileClip, 
                           CompositeAudioClip, ColorClip, CompositeVideoClip, ImageClip)
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_intro_with_opencv(message, emojis, recipient_name, event_type, output_size=(720, 1280), duration=3, fps=30):
    """Create professional intro video with gradients and animations"""
    
    # Professional gradient color schemes (BGR format)
    theme_gradients = {
        'birthday': {
            'top': (220, 180, 255),      # Light purple
            'bottom': (180, 100, 230),   # Deep purple
            'accent': (255, 200, 100),   # Gold
            'text_color': (255, 255, 255)
        },
        'diwali': {
            'top': (50, 180, 255),       # Bright orange
            'bottom': (20, 80, 180),     # Deep red-orange
            'accent': (100, 255, 255),   # Yellow
            'text_color': (255, 255, 255)
        },
        'farewell': {
            'top': (200, 170, 120),      # Light blue
            'bottom': (150, 100, 50),    # Deep blue
            'accent': (255, 200, 150),   # Peach
            'text_color': (255, 255, 255)
        },
        'congratulations': {
            'top': (100, 200, 100),      # Light green
            'bottom': (30, 120, 30),     # Deep green
            'accent': (100, 255, 255),   # Gold
            'text_color': (255, 255, 255)
        },
        'thank_you': {
            'top': (200, 150, 180),      # Light pink
            'bottom': (150, 80, 120),    # Deep pink
            'accent': (255, 180, 200),   # Rose
            'text_color': (255, 255, 255)
        },
        'get_well_soon': {
            'top': (220, 180, 150),      # Light blue
            'bottom': (180, 120, 80),    # Deep blue
            'accent': (150, 255, 150),   # Light green
            'text_color': (255, 255, 255)
        }
    }
    
    colors = theme_gradients.get(event_type, theme_gradients['birthday'])
    frames = []
    total_frames = duration * fps
    
    # Pre-calculate radial gradient ONCE (much faster!)
    center_x, center_y = output_size[0] // 2, output_size[1] // 2
    y_coords, x_coords = np.ogrid[:output_size[1], :output_size[0]]
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    ratios = distances / max_dist
    
    # Create base gradient
    base_gradient = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    for i in range(3):
        base_gradient[:, :, i] = (colors['top'][i] * (1 - ratios) + colors['bottom'][i] * ratios).astype(np.uint8)
    
    # Create vignette ONCE
    vignette = (1 - (ratios * 0.4)).clip(0, 1)
    
    for frame_num in range(total_frames):
        # Animation progress (0 to 1)
        progress = frame_num / total_frames
        
        # Copy base gradient
        frame = base_gradient.copy()
        
        # Apply vignette
        frame = (frame * vignette[:, :, np.newaxis]).astype(np.uint8)
        
        # Add animated particles (fewer, simpler)
        num_particles = 12
        for i in range(num_particles):
            angle = (i * 360 / num_particles + progress * 120) % 360
            radius = 150 + (i % 2) * 100
            px = int(center_x + radius * np.cos(np.radians(angle)))
            py = int(center_y + radius * np.sin(np.radians(angle)))
            
            if 0 <= px < output_size[0] and 0 <= py < output_size[1]:
                particle_size = 20
                alpha = 0.4 + 0.2 * np.sin(progress * np.pi * 2 + i)
                color_val = int(255 * alpha)
                particle_color = tuple(min(255, int(colors['accent'][j] * alpha)) for j in range(3))
                cv2.circle(frame, (px, py), particle_size, particle_color, -1)
                cv2.circle(frame, (px, py), particle_size, (255, 255, 255), 2)
        
        # Add elegant corner decorations
        corner_size = 60
        corner_thick = 5
        # Top-left corner
        cv2.line(frame, (15, 15), (corner_size, 15), colors['accent'], corner_thick)
        cv2.line(frame, (15, 15), (15, corner_size), colors['accent'], corner_thick)
        cv2.circle(frame, (15, 15), 8, colors['accent'], -1)
        # Top-right corner
        cv2.line(frame, (output_size[0]-15, 15), (output_size[0]-corner_size, 15), colors['accent'], corner_thick)
        cv2.line(frame, (output_size[0]-15, 15), (output_size[0]-15, corner_size), colors['accent'], corner_thick)
        cv2.circle(frame, (output_size[0]-15, 15), 8, colors['accent'], -1)
        # Bottom-left corner
        cv2.line(frame, (15, output_size[1]-15), (corner_size, output_size[1]-15), colors['accent'], corner_thick)
        cv2.line(frame, (15, output_size[1]-15), (15, output_size[1]-corner_size), colors['accent'], corner_thick)
        cv2.circle(frame, (15, output_size[1]-15), 8, colors['accent'], -1)
        # Bottom-right corner
        cv2.line(frame, (output_size[0]-15, output_size[1]-15), (output_size[0]-corner_size, output_size[1]-15), colors['accent'], corner_thick)
        cv2.line(frame, (output_size[0]-15, output_size[1]-15), (output_size[0]-15, output_size[1]-corner_size), colors['accent'], corner_thick)
        cv2.circle(frame, (output_size[0]-15, output_size[1]-15), 8, colors['accent'], -1)
        
        # Fade-in animation for text
        text_alpha = min(1.0, progress * 2)  # Fade in during first half
        
        # Add recipient name at top with animation
        if recipient_name:
            text = f"Dear {recipient_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.5
            thickness = 4
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = (output_size[0] - text_width) // 2
            y = 250
            
            # Animated slide-in from top
            y_offset = int((1 - text_alpha) * -50)
            y = y + y_offset
            
            # Glow effect - larger and more visible
            for offset in range(12, 0, -3):
                alpha = 0.4 * text_alpha
                glow_color = tuple(int(c * alpha) for c in colors['accent'])
                cv2.putText(frame, text, (x, y), font, font_scale, glow_color, thickness + offset)
            
            # Strong shadow for depth
            cv2.putText(frame, text, (x+5, y+5), font, font_scale, (0, 0, 0), thickness+2)
            # Main text - bright white
            cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # Add message in center with better styling
        words = message.split()
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_TRIPLEX, 1.8, 3)
            if w < output_size[0] - 120:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        # Center the message block
        y_start = output_size[1] // 2 - (len(lines) * 85) // 2
        for i, line in enumerate(lines):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 4
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            x = (output_size[0] - text_width) // 2
            y = y_start + i * 90
            
            # Animated slide-in from bottom
            y_offset = int((1 - text_alpha) * 30)
            y = y + y_offset
            
            # Glow effect for message - stronger
            for offset in range(10, 0, -2):
                alpha = 0.35 * text_alpha
                glow_color = tuple(int(c * alpha) for c in colors['accent'])
                cv2.putText(frame, line, (x, y), font, font_scale, glow_color, thickness + offset)
            
            # Strong shadow
            cv2.putText(frame, line, (x+4, y+4), font, font_scale, (0, 0, 0), thickness+2)
            # Main text - bright white
            cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # Draw elegant decorative line at bottom instead of emojis
        # Create a decorative separator
        line_y = output_size[1] - 180
        line_start = 100
        line_end = output_size[0] - 100
        
        # Main decorative line
        cv2.line(frame, (line_start, line_y), (line_end, line_y), colors['accent'], 3)
        
        # Add decorative circles along the line
        num_circles = 5
        for i in range(num_circles):
            x_pos = line_start + (line_end - line_start) * i // (num_circles - 1)
            circle_size = 12 if i % 2 == 0 else 8
            cv2.circle(frame, (x_pos, line_y), circle_size, colors['accent'], -1)
            cv2.circle(frame, (x_pos, line_y), circle_size, (255, 255, 255), 2)
        
        # Add small accent lines
        for i in range(3):
            offset = 30 + i * 15
            cv2.line(frame, (line_start - 20, line_y - offset), (line_start + 30, line_y - offset), colors['accent'], 2)
            cv2.line(frame, (line_end - 30, line_y - offset), (line_end + 20, line_y - offset), colors['accent'], 2)
        
        # Convert BGR to RGB for MoviePy
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    # Create video clip from frames
    intro_clip = ImageClip(frames[0], duration=duration)
    intro_clip = intro_clip.fl(lambda gf, t: frames[int(t * fps) if int(t * fps) < len(frames) else -1], apply_to=[])
    
    return intro_clip


def create_border_overlay(size, theme_type, duration):
    """Create a static border overlay image"""
    width, height = size
    border_thickness = 20
    
    # Create transparent image
    border_img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Theme colors (RGBA format)
    theme_colors = {
        'birthday': (255, 105, 180, 255),  # Hot pink
        'diwali': (255, 140, 0, 255),      # Orange
        'farewell': (70, 130, 180, 255),   # Steel blue
        'congratulations': (50, 205, 50, 255),  # Lime green
        'thank_you': (147, 112, 219, 255),      # Purple
        'get_well_soon': (100, 149, 237, 255)   # Cornflower blue
    }
    
    color = theme_colors.get(theme_type, (200, 200, 200, 255))
    
    # Draw outer border
    border_img[:border_thickness, :] = color  # Top
    border_img[-border_thickness:, :] = color  # Bottom
    border_img[:, :border_thickness] = color  # Left
    border_img[:, -border_thickness:] = color  # Right
    
    # Draw inner white border
    inner_thickness = border_thickness // 3
    offset = border_thickness
    white = (255, 255, 255, 255)
    
    border_img[offset:offset+inner_thickness, offset:-offset] = white  # Top inner
    border_img[-offset-inner_thickness:-offset, offset:-offset] = white  # Bottom inner
    border_img[offset:-offset, offset:offset+inner_thickness] = white  # Left inner
    border_img[offset:-offset, -offset-inner_thickness:-offset] = white  # Right inner
    
    # Create ImageClip from the border
    border_clip = ImageClip(border_img, duration=duration, ismask=False)
    return border_clip


def add_border_and_name_opencv(video_clip, name, theme_type):
    """Add professional themed border, name overlay, and floating animated elements"""
    w, h = video_clip.size
    border_thickness = 20
    
    # Professional theme colors matching intro (BGR format)
    theme_colors = {
        'birthday': (200, 150, 255),      # Purple
        'diwali': (50, 180, 255),         # Orange
        'farewell': (180, 140, 90),       # Blue
        'congratulations': (80, 180, 80), # Green
        'thank_you': (200, 130, 160),     # Pink
        'get_well_soon': (200, 160, 120)  # Light blue
    }
    
    border_color = theme_colors.get(theme_type, (200, 200, 200))
    accent_color = tuple(int(c * 1.2) for c in border_color)  # Lighter accent
    
    # Floating shapes configuration based on theme (using OpenCV shapes, not emojis)
    shape_configs = {
        'birthday': {'shape': 'circle', 'color': (180, 105, 255)},  # Pink circles (balloons)
        'diwali': {'shape': 'flame', 'color': (0, 165, 255)},       # Orange flames (diyas)
        'farewell': {'shape': 'heart', 'color': (147, 112, 219)},   # Purple hearts
        'congratulations': {'shape': 'star', 'color': (0, 215, 255)}, # Gold stars
        'thank_you': {'shape': 'heart', 'color': (147, 20, 255)},   # Pink hearts
        'get_well_soon': {'shape': 'circle', 'color': (50, 205, 50)} # Green circles
    }
    
    config = shape_configs.get(theme_type, {'shape': 'star', 'color': (255, 215, 0)})
    
    # Create random positions for floating elements
    np.random.seed(hash(name) % 1000)  # Different seed per video to avoid all starting at same position
    num_elements = 5
    element_data = []
    for i in range(num_elements):
        element_data.append({
            'shape': config['shape'],
            'color': config['color'],
            'x_start': np.random.randint(50, w-100),
            'y_start_offset': np.random.uniform(0, h),  # Start at different heights
            'y_speed': np.random.uniform(50, 120),  # pixels per second
            'x_drift': np.random.uniform(-15, 15),  # horizontal drift
            'size': np.random.randint(15, 30)
        })
    
    def process_frame(get_frame, t):
        frame = get_frame(t)
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw professional double border
        # Outer border
        cv2.rectangle(frame_bgr, (0, 0), (w-1, h-1), border_color, border_thickness)
        
        # Inner accent border
        inner_offset = border_thickness
        inner_thickness = 6
        cv2.rectangle(frame_bgr, (inner_offset, inner_offset), 
                     (w-inner_offset-1, h-inner_offset-1), accent_color, inner_thickness)
        
        # Add corner decorations
        corner_size = 40
        # Top-left
        cv2.line(frame_bgr, (10, 10), (corner_size, 10), (255, 255, 255), 3)
        cv2.line(frame_bgr, (10, 10), (10, corner_size), (255, 255, 255), 3)
        # Top-right
        cv2.line(frame_bgr, (w-10, 10), (w-corner_size, 10), (255, 255, 255), 3)
        cv2.line(frame_bgr, (w-10, 10), (w-10, corner_size), (255, 255, 255), 3)
        # Bottom-left
        cv2.line(frame_bgr, (10, h-10), (corner_size, h-10), (255, 255, 255), 3)
        cv2.line(frame_bgr, (10, h-10), (10, h-corner_size), (255, 255, 255), 3)
        # Bottom-right
        cv2.line(frame_bgr, (w-10, h-10), (w-corner_size, h-10), (255, 255, 255), 3)
        cv2.line(frame_bgr, (w-10, h-10), (w-10, h-corner_size), (255, 255, 255), 3)
        
        # Add floating animated shapes
        for elem in element_data:
            # Calculate position based on time - start from different positions
            y_pos = int((elem['y_start_offset'] + h) - (elem['y_speed'] * t) % (h + 200))
            x_pos = int(elem['x_start'] + np.sin(t * 2) * elem['x_drift'])
            
            if 0 <= y_pos < h and 0 <= x_pos < w:
                size = elem['size']
                color = elem['color']
                
                if elem['shape'] == 'circle':
                    # Draw circle (balloon)
                    cv2.circle(frame_bgr, (x_pos, y_pos), size, color, -1)
                    cv2.circle(frame_bgr, (x_pos, y_pos), size, (255, 255, 255), 2)
                    # Balloon string
                    cv2.line(frame_bgr, (x_pos, y_pos + size), (x_pos, y_pos + size + 30), (255, 255, 255), 2)
                    
                elif elem['shape'] == 'star':
                    # Draw star
                    pts = np.array([
                        [x_pos, y_pos - size],
                        [x_pos + size//3, y_pos - size//3],
                        [x_pos + size, y_pos - size//3],
                        [x_pos + size//2, y_pos + size//4],
                        [x_pos + size*2//3, y_pos + size],
                        [x_pos, y_pos + size//2],
                        [x_pos - size*2//3, y_pos + size],
                        [x_pos - size//2, y_pos + size//4],
                        [x_pos - size, y_pos - size//3],
                        [x_pos - size//3, y_pos - size//3]
                    ], np.int32)
                    cv2.fillPoly(frame_bgr, [pts], color)
                    cv2.polylines(frame_bgr, [pts], True, (255, 255, 255), 2)
                    
                elif elem['shape'] == 'heart':
                    # Draw professional heart shape
                    # Left circle
                    cv2.circle(frame_bgr, (x_pos - size//2, y_pos - size//3), size//2, color, -1)
                    # Right circle
                    cv2.circle(frame_bgr, (x_pos + size//2, y_pos - size//3), size//2, color, -1)
                    # Triangle bottom
                    pts = np.array([[x_pos - size, y_pos - size//3],
                                   [x_pos, y_pos + size],
                                   [x_pos + size, y_pos - size//3]], np.int32)
                    cv2.fillPoly(frame_bgr, [pts], color)
                    # White outline for depth
                    cv2.circle(frame_bgr, (x_pos - size//2, y_pos - size//3), size//2, (255, 255, 255), 2)
                    cv2.circle(frame_bgr, (x_pos + size//2, y_pos - size//3), size//2, (255, 255, 255), 2)
                    
                elif elem['shape'] == 'flame':
                    # Draw flame shape (diya)
                    pts = np.array([
                        [x_pos, y_pos - size],
                        [x_pos + size//2, y_pos],
                        [x_pos, y_pos + size//2],
                        [x_pos - size//2, y_pos]
                    ], np.int32)
                    cv2.fillPoly(frame_bgr, [pts], color)
                    cv2.polylines(frame_bgr, [pts], True, (255, 255, 0), 2)
        
        # Add professional name overlay at bottom
        text = f"From: {name}"
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.4
        thickness = 3
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - text_width) // 2
        y = h - 70
        
        # Semi-transparent background bar
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, h-120), (w, h), (0, 0, 0), -1)
        frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)
        
        # Glow effect for name
        for offset in range(6, 0, -2):
            glow_color = tuple(int(c * 0.3) for c in border_color)
            cv2.putText(frame_bgr, text, (x, y), font, font_scale, glow_color, thickness + offset)
        
        # Shadow
        cv2.putText(frame_bgr, text, (x+3, y+3), font, font_scale, (0, 0, 0), thickness+1)
        # White text
        cv2.putText(frame_bgr, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # Convert back to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    return video_clip.fl(process_frame, apply_to=[])


def compile_videos(campaign, app_config):
    """Main function to compile all videos"""
    print("\n" + "="*50)
    print("Starting video compilation...")
    print("="*50)
    
    app_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(os.path.join(app_dir, app_config['COMPILED_FOLDER']), exist_ok=True)
    
    # Get orientation from campaign (portrait or landscape)
    orientation = campaign.get('orientation', 'portrait')
    
    if orientation == 'landscape':
        output_width = 1280
        output_height = 720
        print("📐 Using LANDSCAPE orientation (1280x720)")
    else:
        output_width = 720
        output_height = 1280
        print("📐 Using PORTRAIT orientation (720x1280)")
    
    # Step 1: Create intro
    print("\n[1/4] Creating intro video...")
    intro_clip = create_intro_with_opencv(
        campaign.get('ai_message', 'Thank you for your message!'),
        campaign.get('ai_emojis', '🎉 ✨ 💝'),
        campaign.get('recipient_name', ''),
        campaign.get('event_type', 'birthday'),
        (output_width, output_height)
    )
    print("✓ Intro created successfully")
    
    # Step 2: Process individual videos
    print(f"\n[2/4] Processing {len(campaign['videos'])} videos...")
    processed_videos = []
    
    for idx, video in enumerate(campaign['videos']):
        input_path = os.path.join(app_dir, video['path'])
        print(f"  Processing {idx + 1}/{len(campaign['videos'])}: {video['name']}")
        
        try:
            # Load original video
            clip = VideoFileClip(input_path)
            
            # Don't resize - keep original aspect ratio and size
            # Just ensure it fits within bounds
            if clip.w > output_width or clip.h > output_height:
                scale = min(output_width / clip.w, output_height / clip.h)
                new_w = int(clip.w * scale)
                new_h = int(clip.h * scale)
                clip = clip.resize((new_w, new_h))
            
            # Add border and name overlay using OpenCV
            clip = add_border_and_name_opencv(clip, video['name'], campaign['event_type'])
            
            processed_videos.append(clip)
            print(f"  ✓ Video {idx + 1} processed")
            
        except Exception as e:
            print(f"  ✗ Error processing video: {str(e)}")
            continue
    
    if not processed_videos:
        raise Exception("No videos could be processed successfully")
    
    print(f"✓ All {len(processed_videos)} videos processed")
    
    # Step 3: Concatenate all clips
    print("\n[3/4] Combining all clips...")
    all_clips = [intro_clip] + processed_videos
    final_video = concatenate_videoclips(all_clips, method="compose")
    print(f"✓ Combined into {final_video.duration:.1f} second video")
    
    # Step 4: Add background music
    print("\n[4/4] Adding background music...")
    event_type = campaign['event_type']
    music_path = os.path.join(app_dir, 'static', 'music', f"{event_type}.wav")
    
    print(f"🎵 Looking for event music: {event_type}.wav")
    
    if not os.path.exists(music_path):
        print(f"⚠️  Event music not found, using default")
        music_path = os.path.join(app_dir, 'static', 'music', 'default_background.wav')
    else:
        print(f"✓ Using event-specific music: {event_type}.wav")
    
    if os.path.exists(music_path):
        try:
            background_music = AudioFileClip(music_path)
            
            # Loop music if needed
            if background_music.duration < final_video.duration:
                n_loops = int(np.ceil(final_video.duration / background_music.duration))
                from moviepy.audio.AudioClip import concatenate_audioclips
                background_music = concatenate_audioclips([background_music] * n_loops)
            
            # Trim to video duration
            background_music = background_music.subclip(0, final_video.duration)
            background_music = background_music.volumex(0.2)  # 20% volume
            
            # Mix with original audio
            if final_video.audio is not None:
                final_audio = CompositeAudioClip([final_video.audio, background_music])
            else:
                final_audio = background_music
            
            final_video = final_video.set_audio(final_audio)
            print("✓ Background music added")
        except Exception as e:
            print(f"⚠ Could not add music: {str(e)}")
    else:
        print("⚠ No background music file found")
    
    # Step 5: Write final video
    print("\n[5/5] Writing final video file...")
    final_output = os.path.join(app_dir, app_config['COMPILED_FOLDER'], 
                               f"{campaign['id']}_final.mp4")
    
    final_video.write_videofile(
        final_output,
        codec='libx264',
        audio_codec='aac',
        fps=30,
        preset='medium',
        threads=4,
        ffmpeg_params=['-crf', '23']
    )
    
    # Cleanup
    print("\n[6/6] Cleaning up...")
    for clip in all_clips:
        clip.close()
    
    print("\n" + "="*50)
    print(f"✓ COMPILATION COMPLETE!")
    print(f"Output: {final_output}")
    print("="*50 + "\n")
    
    return final_output
