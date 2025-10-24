"""
Helper script to create sample music files for different events.
You can replace these with actual music files later.
"""
import os
import numpy as np
from scipy.io import wavfile

def create_sample_music(filename, duration=10, frequency=440, sample_rate=44100):
    """Create a simple tone as placeholder music"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple melody with multiple frequencies
    audio = np.zeros_like(t)
    
    if 'birthday' in filename:
        # Happy, upbeat melody
        freqs = [523, 587, 659, 698, 784]  # C, D, E, F, G
    elif 'diwali' in filename:
        # Festive, energetic melody
        freqs = [440, 494, 554, 587, 659]  # A, B, C#, D, E
    elif 'congratulations' in filename:
        # Triumphant melody
        freqs = [523, 659, 784, 880, 1047]  # C, E, G, A, C
    elif 'farewell' in filename:
        # Gentle, nostalgic melody
        freqs = [392, 440, 494, 523, 587]  # G, A, B, C, D
    else:
        # Default pleasant melody
        freqs = [440, 494, 523, 587, 659]  # A, B, C, D, E
    
    # Create melody
    segment_duration = duration / len(freqs)
    for i, freq in enumerate(freqs):
        start = int(i * segment_duration * sample_rate)
        end = int((i + 1) * segment_duration * sample_rate)
        audio[start:end] = np.sin(2 * np.pi * freq * t[start:end])
    
    # Add fade in/out
    fade_samples = int(0.1 * sample_rate)
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    
    return audio, sample_rate

def main():
    # Create music directory
    music_dir = os.path.join('static', 'music')
    os.makedirs(music_dir, exist_ok=True)
    
    # Event types
    events = [
        'birthday',
        'diwali',
        'farewell',
        'congratulations',
        'thank_you',
        'anniversary',
        'get_well_soon',
        'default_background'
    ]
    
    print("Creating sample music files...")
    print("=" * 50)
    
    for event in events:
        filename = os.path.join(music_dir, f"{event}.wav")
        audio, sample_rate = create_sample_music(filename, duration=10)
        wavfile.write(filename, sample_rate, audio)
        print(f"✓ Created: {filename}")
    
    print("=" * 50)
    print("\n✅ All sample music files created!")
    print("\n📝 Note: These are simple placeholder tones.")
    print("   Replace them with actual music files for better results.")
    print("\n   Recommended: Download royalty-free music from:")
    print("   - Pixabay Music (pixabay.com/music)")
    print("   - YouTube Audio Library")
    print("   - Free Music Archive")

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("⚠️  scipy not installed. Installing...")
        print("   Run: pip install scipy")
        print("\n   Or use this simpler version without scipy:")
        print()
        
        # Fallback without scipy
        import wave
        import struct
        
        music_dir = os.path.join('static', 'music')
        os.makedirs(music_dir, exist_ok=True)
        
        events = ['birthday', 'diwali', 'farewell', 'congratulations', 
                 'thank_you', 'anniversary', 'get_well_soon', 'default_background']
        
        for event in events:
            filename = os.path.join(music_dir, f"{event}.wav")
            
            # Create simple beep
            sample_rate = 44100
            duration = 10
            frequency = 440
            
            with wave.open(filename, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                
                for i in range(int(duration * sample_rate)):
                    value = int(32767 * 0.3 * np.sin(2 * np.pi * frequency * i / sample_rate))
                    wav_file.writeframes(struct.pack('h', value))
            
            print(f"✓ Created: {filename}")
        
        print("\n✅ Basic music files created!")
