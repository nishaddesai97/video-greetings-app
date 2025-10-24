# Background Music Files

## Required Music Files

Place your event-specific music files in this directory with these exact names:

- `birthday.wav` - Upbeat, celebratory music for birthdays
- `diwali.wav` - Festive Indian music for Diwali celebrations
- `farewell.wav` - Gentle, nostalgic music for farewells
- `congratulations.wav` - Triumphant, energetic music for achievements
- `thank_you.wav` - Warm, appreciative music
- `anniversary.wav` - Romantic music for anniversaries
- `get_well_soon.wav` - Uplifting, hopeful music
- `default_background.wav` - Default music for other events

## Format Requirements

- **Format**: WAV (recommended) or MP3
- **Duration**: 10-30 seconds (will auto-loop)
- **Sample Rate**: 44100 Hz recommended
- **Channels**: Stereo or Mono

## Quick Setup

### Option 1: Generate Sample Files (for testing)
```bash
python create_music_files.py
```

### Option 2: Download Royalty-Free Music

**Recommended Sources:**
1. **Pixabay Music** (https://pixabay.com/music/)
   - Free, no attribution required
   - Search for: "happy birthday", "celebration", "farewell"

2. **YouTube Audio Library** (https://studio.youtube.com/channel/UC.../music)
   - Free music for creators
   - Filter by mood and genre

3. **Free Music Archive** (https://freemusicarchive.org/)
   - Creative Commons licensed
   - Wide variety of genres

4. **Incompetech** (https://incompetech.com/music/royalty-free/)
   - Free with attribution
   - Great selection

### Option 3: Convert Your Own Music

If you have MP3 files, convert them to WAV:

**Using FFmpeg:**
```bash
ffmpeg -i your_music.mp3 -acodec pcm_s16le -ar 44100 birthday.wav
```

**Using Online Converter:**
- Visit: https://online-audio-converter.com/
- Upload MP3, select WAV format
- Download and rename

## Tips

1. **Keep it short**: 10-20 seconds is ideal (it will loop automatically)
2. **Volume**: The app sets music to 20% volume automatically
3. **Fade**: Music with fade-in/fade-out sounds more professional
4. **Licensing**: Always use royalty-free or licensed music
5. **Testing**: Test with `default_background.wav` first

## Current Status

Run the app and check the console output:
- `✓ Using event-specific music: birthday.wav` - Music file found
- `⚠️ Event music not found, using default` - Add the specific file
