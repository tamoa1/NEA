"""Example NCHW tensor generator for convolution prototyping.

Functions:
- `make_example_input(shape=(1,3,8,8), seed=None)` -> np.ndarray

Also provides a ready-to-import `EXAMPLE_TENSOR`.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def make_example_input(shape: Tuple[int, int, int, int] = (1, 3, 8, 8), seed: Optional[int] = None, dtype=np.float32) -> np.ndarray:
	"""Return a reproducible example tensor shaped `(N, C, H, W)`.

	Args:
		shape: tuple (N, C, H, W)
		seed: optional RNG seed for reproducibility
		dtype: NumPy dtype for returned array

	Returns:
		NumPy array with values in [0, 1).
	"""
	if seed is not None:
		rng = np.random.RandomState(seed)
		return rng.rand(*shape).astype(dtype)
	return np.random.rand(*shape).astype(dtype)


# A small ready-to-import example tensor you can use directly.
EXAMPLE_TENSOR = make_example_input((1, 3, 8, 8), seed=42)













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
        




def convolutional(in_tensor, weight_tensor, stride):
    """Inputs:
        in_tensor: shape (B_in, C_in, H_in, W_in)
        weight_tensor: shape (C_out, C_in, K_h, K_w)
            c_out: the number of filters. each filter is 3d tensor that applies to the input tensor (every feature map) -> this predicts the number of channels in the output tensor as each filter produces a single feature map
            c_in: the number of channels in the input tensor
        stride: how much to move the kernel each time
            for 3x3 have stride 1, for 5x5 have stride 2, for 7x7 have stride 3

    Returns:
        Output tensor after applying convolution.
    """
    # padding
    pad_tensor = padding(in_tensor, weight_tensor)
    
    B_in, C_in, H_in, W_in = pad_tensor.shape
    C_out, C_in, K_h, K_w = weight_tensor.shape
    
    # Calculate the output tensor shape
    H_out = (H_in - K_h) // stride + 1
    W_out = (W_in - K_w) // stride + 1 

    # empty output tensor
    out_tensor = np.zeros((B_in, C_out, H_out, W_out), dtype=in_tensor.dtype) 

    for b in range(B_in):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    # calculate the start and end positions for the current window
                    h_start = h * stride
                    w_start = w * stride
                    h_end = h_start + K_h
                    w_end = w_start + K_w

                    # dot product between the input tensor and the weight tensor for the current window
                    out_tensor[b, c_out, h, w] = np.sum(
                        pad_tensor[b, :, h_start:h_end, w_start:w_end] * weight_tensor[c_out]
                    )

    return out_tensor


def show_tensor(tensor):
    for h in range(tensor.shape[2]):
        print(tensor[0, 0, h, :])


EXAMPLE_TENSOR = make_example_input((1, 3, 8, 8), seed=42)
print("\n\n\n\nExample input tensor:")
show_tensor(EXAMPLE_TENSOR)


conv_weight = make_example_input((2, 3, 3, 3), seed=42)  # Example weight tensor with 2 filters
conv_stride = 1  # Example stride
print("\nWeight tensor:")
show_tensor(conv_weight)


padded_tensor = padding(EXAMPLE_TENSOR, conv_weight)
print("\nPadded tensor:")
show_tensor(padded_tensor)

convolutional_output = convolutional(EXAMPLE_TENSOR, conv_weight, conv_stride)
print("\nConvoluted output tensor:")
show_tensor(convolutional_output)


"""THIS CODE DOES NOT WORK. IT IS A PROTOTYPE. IT IS NOT OPTIMIZED. IT IS NOT EFFICIENT. IT IS NOT COMPLETE. IT IS NOT TESTED. IT IS NOT PRODUCTION READY. IT IS NOT SAFE. IT IS NOT SECURE. IT IS NOT RELIABLE. IT IS NOT ROBUST. IT IS NOT SCALABLE. IT IS NOT MAINTAINABLE. IT IS NOT DOCUMENTED. IT IS NOT SUPPORTED. IT IS NOT GUARANTEED TO WORK. USE AT YOUR OWN RISK."""