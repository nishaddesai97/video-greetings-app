from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import json
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from pathlib import Path
import requests
import cv2
import numpy as np
import math
from moviepy.editor import VideoFileClip, concatenate_audioclips, CompositeAudioClip

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['COMPILED_FOLDER'] = 'compiled'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['COMPILED_FOLDER'], exist_ok=True)

CAMPAIGNS_FILE = 'campaigns.json'

# Initialize campaigns file
if not os.path.exists(CAMPAIGNS_FILE):
    with open(CAMPAIGNS_FILE, 'w') as f:
        json.dump({}, f)

def load_campaigns():
    with open(CAMPAIGNS_FILE, 'r') as f:
        return json.load(f)

def save_campaigns(campaigns):
    with open(CAMPAIGNS_FILE, 'w') as f:
        json.dump(campaigns, f, indent=2)

def create_event_overlay(frame, event_type, frame_width, frame_height):
    """Create event-specific decorative overlay"""
    overlay = np.zeros_like(frame)
    
    # Create border width
    border_width = int(min(frame_width, frame_height) * 0.15)  # Increased to 15% of smallest dimension
    
    if event_type == 'birthday':
        # Create festive border with alternating colors and increased thickness
        for i in range(0, border_width, 4):  # Increased spacing for more visible pattern
            # Main border
            color = [32, 140, 255] if (i//4) % 2 == 0 else [147, 50, 255]  # Alternate between orange and purple
            cv2.rectangle(overlay, (i, i), (frame_width-i, frame_height-i), color, 4)  # Thicker lines
        
        # Add balloon-like decorations in corners with enhanced visibility
        circle_radius = border_width
        balloon_positions = [
            (circle_radius, circle_radius),  # Top-left
            (frame_width-circle_radius, circle_radius),  # Top-right
            (circle_radius, frame_height-circle_radius),  # Bottom-left
            (frame_width-circle_radius, frame_height-circle_radius)  # Bottom-right
        ]
        
        for x, y in balloon_positions:
            # Main balloon
            cv2.circle(overlay, (x, y), circle_radius, [147, 50, 255], -1)  # Filled balloon
            cv2.circle(overlay, (x, y), circle_radius, [255, 255, 255], 2)  # White outline
            
            # Balloon reflection (highlight)
            highlight_pos = (x - circle_radius//3, y - circle_radius//3)
            cv2.circle(overlay, highlight_pos, circle_radius//4, [255, 255, 255], -1)
            
            # Balloon string with wave pattern
            string_start = (x, y + circle_radius)
            for i in range(4):  # Create wavy string
                pt1 = (x + (10 if i % 2 == 0 else -10), y + circle_radius + i * 15)
                pt2 = (x, y + circle_radius + (i + 1) * 15)
                cv2.line(overlay, pt1, pt2, [255, 255, 255], 2)
    
    elif event_type == 'diwali':
        # Create rangoli-inspired border with more intricate pattern
        num_points = 16
        for r in range(0, border_width, 5):
            points = []
            for i in range(num_points):
                angle = i * (2 * np.pi / num_points)
                x = int(frame_width/2 + r * np.cos(angle))
                y = int(frame_height/2 + r * np.sin(angle))
                points.append([x, y])
            
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Create gradient color based on radius
            color = [
                min(255, r * 2), 
                min(255, 128 + r), 
                min(255, 255 - r/2)
            ]
            cv2.polylines(overlay, [pts], True, color, 2)
        
        # Add enhanced diya elements in corners
        diya_positions = [
            (border_width, border_width),
            (frame_width-border_width, border_width),
            (border_width, frame_height-border_width),
            (frame_width-border_width, frame_height-border_width)
        ]
        
        for x, y in diya_positions:
            # Diya base (larger and more detailed)
            diya_size = border_width
            cv2.ellipse(overlay, (x, y), (diya_size, diya_size//2), 0, 0, 360, [0, 165, 255], -1)
            cv2.ellipse(overlay, (x, y), (diya_size, diya_size//2), 0, 0, 360, [255, 255, 255], 2)
            
            # Flame with gradient effect
            flame_center = (x, y-diya_size//2)
            cv2.circle(overlay, flame_center, diya_size//3, [0, 0, 255], -1)  # Red base
            cv2.circle(overlay, flame_center, diya_size//4, [0, 165, 255], -1)  # Orange middle
            cv2.circle(overlay, (x, y-diya_size//2-5), diya_size//6, [255, 255, 255], -1)  # White core
    
    else:
        # Enhanced elegant border for other events
        for i in range(0, border_width, 2):
            # Create a gradient effect
            alpha = 1.0 - (i / border_width)
            color_value = int(200 * alpha)
            color = [color_value, color_value, color_value]
            
            # Draw both rectangular and diagonal patterns
            cv2.rectangle(overlay, (i, i), (frame_width-i, frame_height-i), color, 2)
            if i % 10 == 0:  # Add diagonal lines every 10 pixels
                pts = np.array([[i, 0], [0, i], [frame_width-i, frame_height], 
                              [frame_width, frame_height-i]], np.int32)
                cv2.polylines(overlay, [pts.reshape((-1, 1, 2))], True, color, 1)
    
    # Add corner decorations for all events
    corner_size = border_width // 2
    corners = [
        ((0, 0), (corner_size, corner_size)),
        ((frame_width-corner_size, 0), (frame_width, corner_size)),
        ((0, frame_height-corner_size), (corner_size, frame_height)),
        ((frame_width-corner_size, frame_height-corner_size), (frame_width, frame_height))
    ]
    
    for (x1, y1), (x2, y2) in corners:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), [255, 255, 255], 2)
    
    return overlay

def get_ai_theme(event_type):
    """Generate theme settings based on event type using AI"""
    try:
        # Using free Hugging Face Inference API for theme generation
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        prompt = f"""Generate video theme settings for a {event_type} video in JSON format with these parameters:
        1. color_scheme: warm/cool/neutral
        2. brightness: value between 0.8 and 1.2
        3. saturation: value between 0.8 and 1.2
        4. mood: happy/emotional/energetic/calm
        5. filter: none/vintage/vibrant/dramatic
        Format the response as valid JSON."""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result[0]['generated_text'] if isinstance(result, list) else result['generated_text']
            
            try:
                # Extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if json_match:
                    theme_settings = json.loads(json_match.group())
                    return theme_settings
            except:
                pass
    except Exception as e:
        print(f"AI Theme API Error: {e}")
    
    # Fallback themes
    fallback_themes = {
        'birthday': {
            "color_scheme": "warm",
            "brightness": 1.1,
            "saturation": 1.1,
            "mood": "happy",
            "filter": "vibrant"
        },
        'farewell': {
            "color_scheme": "cool",
            "brightness": 0.9,
            "saturation": 0.9,
            "mood": "emotional",
            "filter": "vintage"
        },
        'diwali': {
            "color_scheme": "warm",
            "brightness": 1.2,
            "saturation": 1.2,
            "mood": "energetic",
            "filter": "vibrant"
        }
    }
    
    return fallback_themes.get(event_type, fallback_themes['birthday'])

def get_ai_music(event_type, duration):
    """Generate background music using Mubert API (free tier)"""
    try:
        # Using Mubert Text-to-Music API (free tier)
        API_URL = "https://api-b2b.mubert.com/v2/TTM"
        
        # Map event types to music moods
        mood_mapping = {
            'birthday': "happy uplifting celebration",
            'farewell': "emotional touching soft",
            'diwali': "festive celebrating ethnic",
            'anniversary': "romantic lovely peaceful",
            'congratulations': "triumphant success happy",
            'thank_you': "grateful peaceful positive",
            'get_well_soon': "hopeful peaceful calm",
            'custom': "positive inspiring"
        }
        
        mood = mood_mapping.get(event_type, "positive inspiring")
        
        # This is a sample implementation - in production, you'd need to handle Mubert authentication
        # For now, we'll return a path to a default music track based on mood
        return f"static/music/{event_type}_background.mp3"
    except Exception as e:
        print(f"AI Music API Error: {e}")
        return None

def get_ai_text_and_emoji(event_type):
    """Generate AI-powered text and emoji using free Hugging Face API"""
    try:
        # Using free Hugging Face Inference API
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        prompt = f"""Generate a short, heartfelt message (max 15 words) and suggest 3 relevant emojis for a {event_type} video greeting. Format:
Message: [your message]
Emojis: [emoji1 emoji2 emoji3]"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        # Note: For production, get a free token from huggingface.co/settings/tokens
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result[0]['generated_text'] if isinstance(result, list) else result['generated_text']
            
            # Parse the response
            lines = generated_text.split('\n')
            message = "Wishing you joy and happiness!"
            emojis = "🎉 ✨ 🎊"
            
            for line in lines:
                if 'Message:' in line:
                    message = line.split('Message:')[1].strip()
                elif 'Emojis:' in line or 'Emoji:' in line:
                    emojis = line.split(':')[1].strip()
            
            return message, emojis
    except Exception as e:
        print(f"AI API Error: {e}")
    
    # Fallback messages
    fallback_messages = {
        'birthday': ("Wishing you a fantastic birthday! 🎂", "🎂 🎉 🎈"),
        'farewell': ("Best wishes on your new journey! 👋", "👋 🌟 💼"),
        'diwali': ("Happy Diwali! May your life shine bright! 🪔", "🪔 ✨ 🎆"),
        'anniversary': ("Celebrating your special milestone! 💑", "💑 🎊 💖"),
        'congratulations': ("Congratulations on your achievement! 🏆", "🏆 🎉 👏"),
        'thank_you': ("Thank you for everything you do! 🙏", "🙏 💐 😊"),
        'get_well_soon': ("Wishing you a speedy recovery! 💪", "💪 🌸 😊"),
        'custom': ("Sending you warm wishes! 💝", "💝 ✨ 🎊")
    }
    
    return fallback_messages.get(event_type, fallback_messages['custom'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create')
def create_page():
    return render_template('create.html')

@app.route('/api/create-campaign', methods=['POST'])
def create_campaign():
    data = request.json
    campaign_id = str(uuid.uuid4())[:8]
    
    # Get AI-generated text and emojis
    ai_message, ai_emojis = get_ai_text_and_emoji(data['event_type'])
    
    campaigns = load_campaigns()
    campaigns[campaign_id] = {
        'id': campaign_id,
        'title': data['title'],
        'event_type': data['event_type'],
        'recipient_name': data.get('recipient_name', ''),
        'created_at': datetime.now().isoformat(),
        'videos': [],
        'ai_message': ai_message,
        'ai_emojis': ai_emojis,
        'compiled_video': None
    }
    save_campaigns(campaigns)
    
    return jsonify({
        'success': True,
        'campaign_id': campaign_id,
        'record_url': url_for('record_page', campaign_id=campaign_id, _external=True),
        'ai_message': ai_message,
        'ai_emojis': ai_emojis
    })

@app.route('/record/<campaign_id>')
def record_page(campaign_id):
    campaigns = load_campaigns()
    if campaign_id not in campaigns:
        return "Campaign not found", 404
    return render_template('record.html', campaign=campaigns[campaign_id])

@app.route('/api/upload-video/<campaign_id>', methods=['POST'])
def upload_video(campaign_id):
    campaigns = load_campaigns()
    if campaign_id not in campaigns:
        return jsonify({'success': False, 'error': 'Campaign not found'}), 404
    
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file'}), 400
    
    file = request.files['video']
    name = request.form.get('name', 'Anonymous')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    # Create campaign folder
    campaign_folder = os.path.join(app.config['UPLOAD_FOLDER'], campaign_id)
    os.makedirs(campaign_folder, exist_ok=True)
    
    # Save video
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{secure_filename(name)}.webm"
    filepath = os.path.join(campaign_folder, filename)
    file.save(filepath)
    
    # Update campaign
    campaigns[campaign_id]['videos'].append({
        'filename': filename,
        'name': name,
        'uploaded_at': datetime.now().isoformat(),
        'path': filepath
    })
    save_campaigns(campaigns)
    
    return jsonify({'success': True, 'video_count': len(campaigns[campaign_id]['videos'])})

@app.route('/compile/<campaign_id>')
def compile_page(campaign_id):
    campaigns = load_campaigns()
    if campaign_id not in campaigns:
        return "Campaign not found", 404
    return render_template('compile.html', campaign=campaigns[campaign_id])

@app.route('/api/compile-videos/<campaign_id>', methods=['POST'])
def compile_videos(campaign_id):
    campaigns = load_campaigns()
    if campaign_id not in campaigns:
        return jsonify({'success': False, 'error': 'Campaign not found'}), 404
    
    campaign = campaigns[campaign_id]
    
    if len(campaign['videos']) == 0:
        return jsonify({'success': False, 'error': 'No videos to compile'}), 400
    
    try:
        # Get the absolute path to the app directory
        app_dir = os.path.abspath(os.path.dirname(__file__))
        
        # Load and concatenate videos with OpenCV
        print("Loading and processing videos...")
        frames_list = []
        audio_clips = []  # Keep audio separate since OpenCV doesn't handle audio
        
        # Get properties from first video
        first_video = cv2.VideoCapture(os.path.join(app_dir, campaign['videos'][0]['path']))
        frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = first_video.get(cv2.CAP_PROP_FPS)
        first_video.release()
        
        # Process each video
        for video in campaign['videos']:
            input_path = os.path.join(app_dir, video['path'])
            print(f"Processing video: {input_path}")
            
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Video file not found: {input_path}")
            
            # Extract frames
            cap = cv2.VideoCapture(input_path)
            
            # Get AI theme settings
            theme = get_ai_theme(campaign['event_type'])
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame if needed
                if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                    frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Apply theme effects
                # Adjust brightness and saturation
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv = hsv.astype('float32')
                
                # Brightness adjustment
                hsv[:,:,2] = hsv[:,:,2] * theme['brightness']
                hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
                
                # Saturation adjustment
                hsv[:,:,1] = hsv[:,:,1] * theme['saturation']
                hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
                
                hsv = hsv.astype('uint8')
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                # Apply color scheme
                color_overlay = np.ones_like(frame, dtype=np.uint8)
                if theme['color_scheme'] == 'warm':
                    color_overlay[:] = [50, 20, 20]  # BGR format
                    frame = cv2.addWeighted(frame, 0.9, color_overlay, 0.1, 0)
                elif theme['color_scheme'] == 'cool':
                    color_overlay[:] = [20, 20, 50]  # BGR format
                    frame = cv2.addWeighted(frame, 0.9, color_overlay, 0.1, 0)
                
                # Apply filter effects
                if theme['filter'] == 'vintage':
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)
                elif theme['filter'] == 'vibrant':
                    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
                elif theme['filter'] == 'dramatic':
                    frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
                
                frames_list.append(frame)
            cap.release()
            
            # Extract audio using moviepy (we'll only use it for audio)
            clip = VideoFileClip(input_path)
            audio_clips.append(clip.audio)
        
        # Create output directories
        os.makedirs(os.path.join(app_dir, app.config['COMPILED_FOLDER']), exist_ok=True)
        temp_video_path = os.path.join(app_dir, app.config['COMPILED_FOLDER'], f"{campaign_id}_temp.mp4")
        final_output_path = os.path.join(app_dir, app.config['COMPILED_FOLDER'], f"{campaign_id}_final.mp4")
        
        # Write frames to video
        print("Writing combined video...")
        out = cv2.VideoWriter(
            temp_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        
            # Add text overlay and decorations to each frame
        font = cv2.FONT_HERSHEY_DUPLEX
        clean_message = campaign['ai_message'].encode('ascii', 'ignore').decode()
        
        # Get text size for positioning
        font_scale = 1
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(clean_message, font, font_scale, thickness)
        
        # Calculate text position (centered, near bottom)
        text_x = (frame_width - text_width) // 2
        text_y = frame_height - 70  # Move text up to make room for speaker name
        
        frame_count = 0
        total_frames = len(frames_list)
        animation_period = fps * 2  # 2-second animation cycle
        
        for frame in frames_list:
            # Create event-specific overlay
            event_overlay = create_event_overlay(frame, campaign['event_type'], frame_width, frame_height)
            
            # Add animation effect to the overlay with increased visibility
            animation_progress = (frame_count % animation_period) / animation_period
            animation_alpha = 0.4 + 0.3 * abs(math.sin(animation_progress * math.pi))  # Pulsing effect with higher base opacity
            frame = cv2.addWeighted(frame, 1, event_overlay, animation_alpha, 0)
            
            # Add semi-transparent background for text
            text_overlay = frame.copy()
            cv2.rectangle(
                text_overlay,
                (text_x - 10, text_y - text_height - 10),
                (text_width + text_x + 10, text_y + 40),  # Extended for speaker name
                (0, 0, 0),
                -1
            )
            frame = cv2.addWeighted(text_overlay, 0.5, frame, 0.5, 0)
            
            # Add main message
            cv2.putText(
                frame,
                clean_message,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
            
            # Get current video's speaker name
            current_video_index = frame_count * len(campaign['videos']) // total_frames
            speaker_name = campaign['videos'][current_video_index]['name']
            
            # Add speaker name below message
            speaker_text = f"From: {speaker_name}"
            (speaker_width, _), _ = cv2.getTextSize(speaker_text, font, 0.7, 1)
            speaker_x = (frame_width - speaker_width) // 2
            cv2.putText(
                frame,
                speaker_text,
                (speaker_x, text_y + 30),  # Position below main message
                font,
                0.7,  # Smaller font size for speaker name
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            frame_count += 1
            out.write(frame)
        
        out.release()
        
        # Combine audio tracks
        print("Adding audio...")
        if audio_clips:
            final_audio = concatenate_audioclips(audio_clips)
            
            # Get AI-generated background music
            total_duration = sum(clip.duration for clip in audio_clips if clip)
            bg_music_path = get_ai_music(campaign['event_type'], total_duration)
            
            if bg_music_path and os.path.exists(bg_music_path):
                try:
                    bg_music = VideoFileClip(bg_music_path).audio
                    bg_music = bg_music.set_duration(total_duration)
                    bg_music = bg_music.volumex(0.3)  # Set background music volume to 30%
                    final_audio = CompositeAudioClip([final_audio, bg_music])
                except Exception as e:
                    print(f"Could not add background music: {e}")
            
            # Load the video we just created
            video = VideoFileClip(temp_video_path)
            
            # Add audio and write final video
            final_video = video.set_audio(final_audio)
            final_video.write_videofile(
                final_output_path,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Clean up
            video.close()
            final_video.close()
            for clip in audio_clips:
                if clip:
                    clip.close()
            
            # Remove temporary video
            os.remove(temp_video_path)
        else:
            # If no audio, just rename the temp file
            os.rename(temp_video_path, final_output_path)
        
        # Update campaign
        campaigns[campaign_id]['compiled_video'] = f"{campaign_id}_final.mp4"
        save_campaigns(campaigns)
        
        return jsonify({
            'success': True,
            'download_url': url_for('download_video', campaign_id=campaign_id, _external=True)
        })
    
    except Exception as e:
        print(f"Error compiling videos: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download/<campaign_id>')
def download_video(campaign_id):
    campaigns = load_campaigns()
    if campaign_id not in campaigns:
        return "Campaign not found", 404
    
    campaign = campaigns[campaign_id]
    if not campaign.get('compiled_video'):
        return "Video not compiled yet", 404
    
    video_path = os.path.join(app.config['COMPILED_FOLDER'], f"{campaign_id}_final.mp4")
    
    if not os.path.exists(video_path):
        return "Video file not found", 404
    
    return send_file(video_path, as_attachment=True, download_name=f"{campaign['title']}_greeting.mp4")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)