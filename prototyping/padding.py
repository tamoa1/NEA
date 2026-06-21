import numpy as np

def padding(in_tensor, window_tensor):

    win_size = window_tensor.shape[2]  # Assuming window_tensor is a sqaure
    if win_size <= 0:
        return in_tensor

    b, c, h, w = in_tensor.shape

    # How many additional pixels needed on each axis to make the input divisible by the window size
    h_need = (win_size - (h % win_size)) % win_size
    w_need = (win_size - (w % win_size)) % win_size

    if h_need == 0 and w_need == 0:
        return in_tensor

    # Split required padding into each side for each axis
    h_before = h_need // 2
    h_after = h_need - h_before
    w_before = w_need // 2
    w_after = w_need - w_before

    # padding function from numpy
    return np.pad(in_tensor, ((0, 0), (0, 0), (h_before, h_after), (w_before, w_after)), mode='constant', constant_values=0)