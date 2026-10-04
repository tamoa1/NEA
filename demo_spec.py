import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
import musdb

def load_audio(file_path, sr=44100):
    """load a mp3 file and return the audio signal and sample rate"""
    audio, sr = lr.load(file_path, sr=sr)
    return audio, sr


def STFT(audio, n_fft=2048, hop_length=512):
    """turn into spectrogram"""
    stft = lr.stft(audio, n_fft=n_fft, hop_length=hop_length)
    return stft


def plot_spectrogram(stft, sr=44100, hop_length=512):
    """plot the spectrogram of the audio signal"""

    magnitude = np.abs(stft)
    magnitude_db = lr.amplitude_to_db(magnitude, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(magnitude_db, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.tight_layout()
    plt.show()



#show input and show spectrogram of input 

song, sr = load_audio("C:/Users/toma/OneDrive/Documents/visual studio code/NEA/The Prodigy - Breathe.mp3")
stft = STFT(song)
plot_spectrogram(stft)