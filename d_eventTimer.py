from contextlib import contextmanager
import resource
import gc
import torch


class EventTimer:
    def __init__(self, device):
        self.device = device
        # Warm-up GPU
        torch.randn(3, 3, device=device) @ torch.randn(3, 3, device=device)
        torch.cuda.empty_cache()
        gc.collect()
        self.reset()

    def reset(self):
        """Reset the timer"""
        self.initialized_keys = set()
        self.time_data = dict()  # the time for each occurence of each event
        self.cuda_max_mem_data = dict()
        self.cuda_allocated_mem_data = dict()
        self.ram_allocated_mem_data = dict()

    def create_label_if_not_exists(self, label):
        # Update first and last occurrence of this label
        if label not in self.initialized_keys:
            self.time_data[label] = []
            self.cuda_max_mem_data[label] = []
            self.cuda_allocated_mem_data[label] = []
            self.ram_allocated_mem_data[label] = []
            self.initialized_keys.add(label)

    @contextmanager
    def __call__(self, label):
        # Wait for everything before me to finish
        torch.cuda.current_stream().synchronize()

        # Measure the time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        cuda_mem_offset = torch.cuda.memory_allocated()
        mem_offset = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start.record()
        yield
        # Wait for operations that happen during yield to finish
        torch.cuda.current_stream().synchronize()
        end.record()

        # Need to wait once more for operations to finish
        torch.cuda.current_stream().synchronize()
        self.create_label_if_not_exists(label)

        self.time_data[label].append(start.elapsed_time(end) / 1000)  # seconds

        self.cuda_max_mem_data[label].append(
            (torch.cuda.max_memory_allocated() - cuda_mem_offset) / (1024 * 1024))  # MiB
        self.cuda_allocated_mem_data[label].append(
            (torch.cuda.memory_allocated() - cuda_mem_offset) / (1024 * 1024)
        )
        self.ram_allocated_mem_data[label].append(
            (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - mem_offset) / 1024
        )  # MiB

        # torch.cuda.reset_max_memory_allocated()
        # torch.cuda.reset_peak_memory_stats()

    def summary(self):
        return {
            'time': {
                k: torch.tensor(v) for k, v in self.time_data.items()
            },
            'cuda-max': {
                k: torch.tensor(v) for k, v in self.cuda_max_mem_data.items()
            },
            'cuda-current': {
                k: torch.tensor(v) for k, v in self.cuda_allocated_mem_data.items()
            },
            'ram': {
                k: torch.tensor(v) for k, v in self.ram_allocated_mem_data.items()
            }
        }

    def save_results(self, addr):
        ret = self.summary()
        torch.save(ret, addr)
