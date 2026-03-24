import atexit
import os
import time
from collections import Counter

import pandas as pd
import psutil
import torch
from tabulate import tabulate


class Profiler:
    """A Profiler class to track execution time and memory usage of code blocks."""

    def __init__(self, save_path=None):
        """Initialize the parent Profiler object to track multiple keys."""
        self.metrics = []
        self.counter = Counter()
        self.process = psutil.Process(os.getpid())
        self.save_path = save_path
        atexit.register(self.report)

    def record(self, key):
        """Create a context manager for tracking a specific key.

        Args:
        - key (str): A label for the profiled block.

        Returns:
        - A SimpleProfiler context that tracks runtime and memory usage.
        """
        return SimpleProfiler(key, self)

    def add_metrics(
        self,
        key,
        elapsed_time,
        memory_change,
        end_mem,
        reserve_change,
        reserved_mem,
        cpu_memory_change,
        end_cpu_mem,
    ):
        """Add metrics to the Profiler for a specific key.

        Args:
          key (str): The key associated with the profiled block.
          elapsed_time (float): The time taken for the block.
          memory_change (int): The change in CUDA memory (in bytes).
          end_mem (int): The total CUDA memory at the end of the block.
          reserve_change (int): The change in reserved CUDA memory (in bytes).
          reserved_mem (int): The total reserved CUDA memory at the end of the block.
          cpu_memory_change (int): The change in CPU memory (in bytes).
          end_cpu_mem (int): The total CPU memory at the end of the block.
        """
        self.counter.update([key])
        n_calls = self.counter[key]
        self.metrics.append(
            {
                "key": key,
                "elapsed_time (s)": elapsed_time,
                "CUDA_delta_memory (MB)": memory_change / 1e6,
                "CUDA_memory (MB)": end_mem / 1e6,
                "CUDA_delta_reserved (MB)": memory_change / 1e6,
                "CUDA_reserved (MB)": reserved_mem / 1e6,
                "CPU_delta_memory (MB)": cpu_memory_change / (1024**2),
                "CPU_memory (MB)": end_cpu_mem / (1024**2),
                "calls": n_calls,
            }
        )

    def report(self, average=True):
        """Print a summary of all collected metrics."""
        df = pd.DataFrame(self.metrics)
        if df.empty:
            return
        elif average:
            average_df = (
                df.groupby("key", sort=False)
                .agg(
                    {
                        "elapsed_time (s)": "mean",
                        "CUDA_delta_memory (MB)": "max",
                        "CUDA_memory (MB)": "max",
                        "CUDA_delta_reserved (MB)": "max",
                        "CUDA_reserved (MB)": "max",
                        "CPU_delta_memory (MB)": "max",
                        "CPU_memory (MB)": "max",
                        "calls": "max",
                    }
                )
                .reset_index()
            )
            print(tabulate(average_df, headers="keys"))
            if self.save_path:
                average_df.to_csv(self.save_path, index=False)
        else:
            print(tabulate(df, headers="keys"))
            if self.save_path:
                df.to_csv(self.save_path, index=False)


class SimpleProfiler:
    """A context manager to track execution time and memory usage for a specific code block."""

    def __init__(self, key, parent_profiler):
        """Initialize the context manager for a specific key.

        Args:
          key (str): The label for the profiled block.
          parent_profiler (Profiler): The parent profiler to store metrics.
        """
        self.key = key
        self.parent_profiler = parent_profiler
        self.process = self.parent_profiler.process
        self.start_time = None
        self.start_mem = None
        self.start_res = None

    def __enter__(self):
        """Start tracking time and CUDA memory."""
        self.start_time = time.time()
        self.start_cpu_mem = self.process.memory_info().rss
        if torch.cuda.is_available():
            self.start_mem = torch.cuda.memory_allocated()
            self.start_res = torch.cuda.memory_reserved()
        else:
            self.start_mem = 0
            self.start_res = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """End tracking time and CUDA memory."""
        end_time = time.time()
        end_cpu_mem = self.process.memory_info().rss
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated()
            reserved_mem = torch.cuda.memory_reserved()
        else:
            end_mem = 0
            reserved_mem = 0

        elapsed_time = end_time - self.start_time
        memory_change = end_mem - self.start_mem
        reserve_change = reserved_mem - self.start_res
        cpu_memory_change = end_cpu_mem - self.start_cpu_mem

        # Store metrics in the parent profiler
        self.parent_profiler.add_metrics(
            self.key,
            elapsed_time,
            memory_change,
            end_mem,
            reserve_change,
            reserved_mem,
            cpu_memory_change,
            end_cpu_mem,
        )
