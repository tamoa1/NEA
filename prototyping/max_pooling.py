from util_functions import make_example_input, show_tensor
from padding import padding
import numpy as np

def max_pool(in_tensor, pool_size):
    padded_tensor = padding(in_tensor, pool_size)
    batch_size, channels, height, width = padded_tensor.shape

# calculate ouptput tensor dimensions
    nheight = height // pool_size
    nwidth = width // pool_size
    out_tensor = np.zeros((batch_size, channels, nheight, nwidth))

    for b in range(batch_size):
        for c in range(channels):
# scanning the batches and each of the channels
            for row in range(0, height - 1, pool_size):
                for col in range(0, width - 1, pool_size):
# moving across each map in pool size steps
                    window = padded_tensor[b, c, row : row + pool_size, col : col + pool_size]
# making a window of the dimentions of the pool
                    max = np.max(window)
                    out_tensor[b,c, row // pool_size, col // pool_size] = max
# finding the max value in the window and placing it in the output tensor
    return out_tensor

