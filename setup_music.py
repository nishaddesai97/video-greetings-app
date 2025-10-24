import os
import wave
import struct

def create_default_wav():
    """Create a simple default background music (a gentle beep)"""
    music_dir = os.path.join('static', 'music')
    os.makedirs(music_dir, exist_ok=True)
    
    default_music = os.path.join(music_dir, 'default_background.wav')
    
    # Create a simple WAV file
    sampleRate = 44100.0
    duration = 1.0
    frequency = 440.0
    
    with wave.open(default_music, 'w') as wavefile:
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(sampleRate)
        
        for i in range(int(duration * sampleRate)):
            value = int(32767.0 * float(i) / float(sampleRate))
            data = struct.pack('<h', value)
            wavefile.writeframes(data)

if __name__ == "__main__":
    create_default_wav()
