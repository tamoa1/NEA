import musdb
import numpy as np
import tensorflow as tf


def load_data(set, chunk_len, n_FFT, hop_length):
    """inp: 
    set: musdb subset to load (train, test, validation)
    chunk_len: length of each chunk in seconds
    n_FFT: number of FFT bins for STFT (2048 is common)
    hop_length: hop length for STFT (512 is common with 2048 n_FFT for 75% overlap of windows)
    out:
    X: array of original audio chunks
    Y: array of corresponding drum audio chunks"""

    set = musdb.DB(root = "data/musdb18", subsets=set)

    chunk_size = 44100 * chunk_len                      # chunk_len seconds at 44100 Hz sampling rate

    X_spec = []
    Y_spec = []
    

    for track in set:
        mix = track.audio.mean(axis=1)                               #orignal audio
        drums = track.targets['drums'].audio.mean(axis=1)            #target drum audio

        num_chunks = mix.shape[0] // chunk_size         #finding number of chunks in the audio

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size                    #defining the start and end of each chunk

            x_chunk = mix[start:end]
            y_chunk = drums[start:end]                  #defining the chunks to be transformed into STFT

            X_spec.append(stft(x_chunk, n_FFT, hop_length))
            Y_spec.append(stft(y_chunk, n_FFT, hop_length))         #appending the STFT of the chunks to the respective arrays




    return np.array(X_spec), np.array(Y_spec)








def stft(x, n_FFT, hop_length):
    """inp:
    x: input audio signal
    n_FFT: number of FFT bins for STFT (2048 is common)
    hop_length: hop length for STFT (512 is common with 2048 n_FFT for 75% overlap of windows)
    out:
    X: STFT of the input audio signal
    """

    stft = tf.signal.stft(x, frame_length=n_FFT, frame_step=hop_length, fft_length=n_FFT)                       #tensorflow stft function
    magnitude = tf.abs(stft)                                                                                    #finding the magnitude of the complex stft output (what we are actually interested in)
    return magnitude