import numpy as np
from typing import Optional, Tuple
from click import Tuple


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


def show_tensor(tensor):
    for h in range(tensor.shape[2]):
        print(tensor[0, 0, h, :])