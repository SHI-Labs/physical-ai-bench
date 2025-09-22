"""Utilities for dealing with cattrs."""

import cattrs
import numpy as np


def get_custom_cattrs_converter() -> cattrs.Converter:
    """
    Singleton function to get a global cattrs converter instance with custom hooks for numpy arrays.
    We leave numpy arrays as they when structuring/unstructuring. This is needed as cattrs cannot handle
    numpy arrays.

    Returns:
        cattr.GenConverter: The singleton cattrs converter instance.
    """
    if not hasattr(get_custom_cattrs_converter, "_converter"):
        converter = cattrs.GenConverter()
        # Structure hook for NumPy arrays which just leaves them as is.
        converter.register_structure_hook(np.ndarray, lambda d, t: d)
        get_custom_cattrs_converter._converter = converter

    return get_custom_cattrs_converter._converter
