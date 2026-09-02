import musdb
import numpy as np

def load_data(set):
    set = musdb.DB(root = "data/musdb18", subsets=set)

    X = []
    Y = []

    for track in set:
        mix = track.audio
        drums = track.targets['drums'].audio
        X.append(mix)
        Y.append(drums)

    return np.array(X), np.array(Y)

X, Y = load_data("train")
