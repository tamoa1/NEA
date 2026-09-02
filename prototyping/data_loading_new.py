import musdb
import numpy as np

def load_data(set, chunk_len):
    """inp: 
    set: musdb subset to load (train, test, validation)
    chunk_len: length of each chunk in seconds
    out:
    X: array of original audio chunks
    Y: array of corresponding drum audio chunks"""

    set = musdb.DB(root = "data/musdb18", subsets=set)

    chunk_size = 44100 * chunk_len                      # chunk_len seconds at 44100 Hz sampling rate

    X = []
    Y = []

    for track in set:
        mix = track.audio                               #orignal audio
        drums = track.targets['drums'].audio            #target drum audio

        num_chunks = mix.shape[0] // chunk_size         #finding number of chunks in the audio

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size                    #defining the start and end of each chunk
            X.append(mix[start:end])
            Y.append(drums[start:end])                  #appending the chunks to the respective arrays


    return np.array(X), np.array(Y)


X, Y = load_data("train", chunk_len=5)
