import os
from moviepy.editor import (VideoFileClip, concatenate_videoclips, AudioFileClip, 
                           CompositeAudioClip, ColorClip, TextClip, CompositeVideoClip,
                           ImageClip)
import numpy as np

def create_intro_video(message, emojis, recipient_name, event_type, output_size=(720, 1280), duration=3):
    """Create intro video with recipient name and message"""
    # Theme colors
    theme_colors = {
        'birthday': (255, 182, 193),
        'diwali': (255, 140, 0),
        'farewell': (70, 130, 180),
        'congratulations': (50, 205, 50),
        'thank_you': (147, 112, 219),
        'get_well_soon': (100, 149, 237)
    }
    
    bg_color = theme_colors.get(event_type, (100, 100, 150))
    bg = ColorClip(size=output_size, color=bg_color, duration=duration)
    
    try:
        clips = [bg]
        
        # Recipient name at top
        if recipient_name:
            txt_recipient = TextClip(
                f"Dear {recipient_name}",
                fontsize=60,
                color='white',
                font='Arial-Bold'
            )
            txt_recipient = txt_recipient.set_position(('center', 250)).set_duration(duration)
            clips.append(txt_recipient)
        
        # Message in center
        txt_message = TextClip(
            message,
            fontsize=50,
            color='white',
            font='Arial-Bold',
            size=(output_size[0]-100, None),
            method='caption'
        )
        txt_message = txt_message.set_position('center').set_duration(duration)
        clips.append(txt_message)
        
        # Emojis at bottom
        txt_emojis = TextClip(
            emojis,
            fontsize=80,
            color='white',
            font='Arial'
        )
        txt_emojis = txt_emojis.set_position(('center', 950)).set_duration(duration)
        clips.append(txt_emojis)
        
        intro = CompositeVideoClip(clips, size=output_size)
        return intro
    except Exception as e:
        print(f"Error creating intro: {e}")
        return bg


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


def add_name_and_border(video_clip, name, theme_type):
    """Add name overlay and themed border to video"""
    try:
        # Create border overlay
        border = create_border_overlay(video_clip.size, theme_type, video_clip.duration)
        
        # Create name text
        name_txt = TextClip(
            f"From: {name}",
            fontsize=45,
            color='white',
            font='Arial-Bold',
            stroke_color='black',
            stroke_width=3
        )
        name_txt = name_txt.set_position(('center', video_clip.h - 100)).set_duration(video_clip.duration)
        
        # Composite: video + border + name
        final_clip = CompositeVideoClip([video_clip, border, name_txt], size=video_clip.size)
        return final_clip
    except Exception as e:
        print(f"Error adding overlay: {e}")
        return video_clip


def compile_videos(campaign, app_config):
    """Main function to compile all videos"""
    print("\n" + "="*50)
    print("Starting video compilation...")
    print("="*50)
    
    app_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(os.path.join(app_dir, app_config['COMPILED_FOLDER']), exist_ok=True)
    
    output_width = 720
    output_height = 1280
    
    # Step 1: Create intro
    print("\n[1/4] Creating intro video...")
    intro_clip = create_intro_video(
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
            
            # Resize to fit dimensions while maintaining aspect ratio
            if clip.w > output_width or clip.h > output_height:
                clip = clip.resize(height=output_height if clip.h > clip.w else None,
                                 width=output_width if clip.w >= clip.h else None)
            
            # Add border and name overlay
            clip = add_name_and_border(clip, video['name'], campaign['event_type'])
            
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
    music_path = os.path.join(app_dir, 'static', 'music', f"{campaign['event_type']}.wav")
    if not os.path.exists(music_path):
        music_path = os.path.join(app_dir, 'static', 'music', 'default_background.wav')
    
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
