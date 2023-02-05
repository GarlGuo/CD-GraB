# -*- coding: utf-8 -*-
import time
from contextlib import contextmanager
from io import StringIO

import numpy as np
import torch


class EventTimer:
    """
    Timer for PyTorch code, measured in milliseconds
    Comes in the form of a contextmanager:

    Example:
    >>> timer = Timer()
    ... for i in range(10):
    ...     with timer("expensive operation"):
    ...         x = torch.randn(100)
    ... print(timer.summary())
    """

    def __init__(self, device):
        self.device = device
        # Warm-up GPU
        torch.randn(3,3, device=device) @ torch.randn(3,3, device=device)
        self.reset()

    def reset(self):
        """Reset the timer"""
        self.time_data = {} # the time for each occurence of each event

    @contextmanager
    def __call__(self, label):
        torch.cuda.current_stream().synchronize() # Wait for everything before me to finish

        # Measure the time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        yield
        torch.cuda.current_stream().synchronize() # Wait for operations that happen during yield to finish
        end.record()
        
        torch.cuda.current_stream().synchronize() # Need to wait once more for operations to finish

        # Update first and last occurrence of this label
        if label not in self.time_data:
            self.time_data[label] = []
        self.time_data[label].append(start.elapsed_time(end))

    def save_results(self, addr):
        torch.save(self.time_data, addr)
