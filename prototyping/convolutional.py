from __future__ import annotations
from util_functions import make_example_input, show_tensor
from padding import padding
import numpy as np
from typing import Optional, Tuple


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