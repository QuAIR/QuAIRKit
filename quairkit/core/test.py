# !/usr/bin/env python3
# Copyright (c) 2025 QuAIR team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
    Common settings and utilities for the unit test
"""

import contextlib
import functools
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil
import torch

TOL32 = 1e-6
TOL64 = 1e-10

REPEAT_ITR = 10


def get_test_device_list() -> List[str]:
    r"""Return the device list for tests.

    Policy:
    - Always test CPU.
    - By default, do NOT expand to CUDA for the whole suite (many tests rely on CPU-only numpy/scipy paths).
      You can opt-in by setting environment variable `QUAIRKIT_TEST_ALL_DEVICES=1`.
    """
    devices = ["cpu"]
    if os.getenv("QUAIRKIT_TEST_ALL_DEVICES", "0") == "1" and torch.cuda.is_available():
        devices.append("cuda")
    return devices


def get_state_simulator_test_device_list() -> List[str]:
    r"""Return the device list for StateSimulator-focused tests.

    Policy:
    - Always test CPU.
    - Automatically include CUDA when available.
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices



class PerformanceMonitor:
    r"""Monitor performance metrics (time and memory) for test functions."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: Optional[float] = None
        self.process: Optional[psutil.Process] = None
        self.cuda_peak_memory: Optional[float] = None
        
    def __enter__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        
        self.peak_memory = self.process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            self.cuda_peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return False
    
    @property
    def wall_time(self) -> float:
        r"""Wall time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def memory_peak(self) -> float:
        r"""Peak memory usage in MB."""
        return self.peak_memory if self.peak_memory is not None else 0.0
    
    @property
    def memory_delta(self) -> float:
        r"""Memory delta from start to peak in MB."""
        if self.start_memory is None or self.peak_memory is None:
            return 0.0
        return self.peak_memory - self.start_memory
    
    def to_dict(self) -> Dict[str, Any]:
        r"""Convert to dictionary for JSON serialization."""
        return {
            "wall_time": self.wall_time,
            "memory_peak_mb": self.memory_peak,
            "memory_delta_mb": self.memory_delta,
            "cuda_peak_memory_mb": self.cuda_peak_memory if self.cuda_peak_memory is not None else 0.0,
        }


def get_version() -> str:
    r"""Get the version from quairkit package."""
    try:
        import quairkit
        return quairkit.__version__
    except (ImportError, AttributeError):
        try:
            import tomli
            pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
            with open(pyproject_path, "rb") as f:
                data = tomli.load(f)
                return data.get("project", {}).get("version", "unknown")
        except Exception:
            return "unknown"


def get_commit_hash() -> str:
    r"""Get the git commit hash of the current repository.
    
    Returns:
        Short commit hash (7 characters) if git is available, otherwise "unknown".
    """
    try:
        import subprocess
        repo_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown"
    except Exception:
        return "unknown"



def generate_test_report(
    test_results: Dict[str, Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    format: str = "json"
) -> Path:
    r"""Generate a test performance report.
    
    Args:
        test_results: Dictionary mapping test names to their performance metrics.
        output_dir: Directory to save the report. Defaults to `tests/reports/`.
        format: Output format, either "json" or "csv". Defaults to "json".
    
    Returns:
        Path to the generated report file.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "tests" / "reports"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    commit_hash = get_commit_hash()
    filename = f"test_performance_commit{commit_hash}.{format}"
    output_path = output_dir / filename
    
    tests_by_file: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for test_name, metrics in test_results.items():
        file_name = metrics.get('file_name', None)
        
        if not file_name or file_name == 'unknown':
            file_key = test_name.split('[')[0] if '[' in test_name else test_name
            parts = file_key.split('_')
            if len(parts) >= 2 and parts[0] == 'test':
                file_name = '_'.join(parts[:2]) + '.py'
            else:
                file_name = file_key + '.py'
        
        if file_name not in tests_by_file:
            tests_by_file[file_name] = {}
        tests_by_file[file_name][test_name] = metrics
    
    file_summaries: Dict[str, Dict[str, Any]] = {}
    for file_name, file_tests in tests_by_file.items():
        file_summaries[file_name] = {
            "total_tests": len(file_tests),
            "total_wall_time": sum(r.get("wall_time", 0.0) for r in file_tests.values()),
            "avg_wall_time": sum(r.get("wall_time", 0.0) for r in file_tests.values()) / len(file_tests) if file_tests else 0.0,
            "max_memory_peak_mb": max((r.get("memory_peak_mb", 0.0) for r in file_tests.values()), default=0.0),
            "max_cuda_memory_mb": max((r.get("cuda_peak_memory_mb", 0.0) for r in file_tests.values()), default=0.0),
        }
    
    if format == "json":
        version = get_version()
        report_data = {
            "version": version,
            "commit_hash": commit_hash,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": test_results,
            "tests_by_file": tests_by_file,
            "file_summaries": file_summaries,
            "summary": {
                "total_tests": len(test_results),
                "total_wall_time": sum(r.get("wall_time", 0.0) for r in test_results.values()),
                "avg_wall_time": sum(r.get("wall_time", 0.0) for r in test_results.values()) / len(test_results) if test_results else 0.0,
                "max_memory_peak_mb": max((r.get("memory_peak_mb", 0.0) for r in test_results.values()), default=0.0),
                "max_cuda_memory_mb": max((r.get("cuda_peak_memory_mb", 0.0) for r in test_results.values()), default=0.0),
            }
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    elif format == "csv":
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["test_name", "file_name", "wall_time", "memory_peak_mb", "memory_delta_mb", "cuda_peak_memory_mb"])
            for test_name, metrics in test_results.items():
                file_name = "unknown"
                for fname, file_tests in tests_by_file.items():
                    if test_name in file_tests:
                        file_name = fname
                        break
                writer.writerow([
                    test_name,
                    file_name,
                    metrics.get("wall_time", 0.0),
                    metrics.get("memory_peak_mb", 0.0),
                    metrics.get("memory_delta_mb", 0.0),
                    metrics.get("cuda_peak_memory_mb", 0.0),
                ])
    
    return output_path



def generate_system_index_permutations(
    num_systems: int,
    num_acted: int,
    max_permutations: Optional[int] = None,
    include_sequential: bool = True,
    include_shuffled: bool = True,
    include_partial: bool = True
) -> List[List[int]]:
    r"""Generate various system index permutations for testing.
    
    Args:
        num_systems: Total number of systems.
        num_acted: Number of systems the operation acts on.
        max_permutations: Maximum number of permutations to generate. If None, generates all.
        include_sequential: Include sequential indices like [0, 1, 2].
        include_shuffled: Include shuffled indices.
        include_partial: Include cases where not all systems are used.
    
    Returns:
        List of system index lists.
    """
    permutations = []
    
    if num_acted > num_systems:
        return []
    
    if include_sequential:
        for start in range(num_systems - num_acted + 1):
            indices = list(range(start, start + num_acted))
            if indices not in permutations:
                permutations.append(indices)
    
    if include_shuffled:
        all_indices = list(range(num_systems))
        for perm in itertools.permutations(all_indices, num_acted):
            perm_list = list(perm)
            if perm_list not in permutations:
                permutations.append(perm_list)
    
    if include_partial:
        for subset_size in range(num_acted, min(num_systems, num_acted + 2) + 1):
            if subset_size <= num_systems:
                for perm in itertools.permutations(range(num_systems), subset_size):
                    if len(perm) == num_acted:
                        perm_list = list(perm)
                        if perm_list not in permutations:
                            permutations.append(perm_list)
    
    if max_permutations is not None and len(permutations) > max_permutations:
        if include_sequential:
            sequential = [p for p in permutations if p == sorted(p)]
            shuffled = [p for p in permutations if p not in sequential]
            permutations = sequential + shuffled[:max_permutations - len(sequential)]
        else:
            permutations = permutations[:max_permutations]
    
    return permutations


def generate_batch_dimensions(
    default_range: Tuple[int, int] = (1, 3),
    special_range: Optional[Tuple[int, int]] = None
) -> List[List[int]]:
    r"""Generate batch dimension combinations for testing.
    
    Args:
        default_range: Default range for batch sizes (min, max).
        special_range: Special range for cases needing more systems (min, max).
    
    Returns:
        List of batch dimension lists, including [], [1], [n], [m, n] combinations.
    """
    batch_dims = []
    
    batch_dims.append([])
    batch_dims.append([1])
    
    min_batch, max_batch = default_range
    for n in range(min_batch, max_batch + 1):
        batch_dims.append([n])
    
    for m in range(min_batch, min(max_batch + 1, 3)):
        for n in range(min_batch, min(max_batch + 1, 3)):
            batch_dims.append([m, n])
    
    if special_range is not None:
        min_special, max_special = special_range
        for n in range(min_special, max_special + 1):
            if [n] not in batch_dims:
                batch_dims.append([n])
    
    return batch_dims



def generate_system_dim_combinations(
    num_systems: int,
    default_dim_range: Tuple[int, int] = (2, 3),
    qudit_combinations: Optional[List[List[int]]] = None
) -> List[List[int]]:
    r"""Generate system dimension combinations for testing.
    
    Args:
        num_systems: Number of systems.
        default_dim_range: Default dimension range (min, max).
        qudit_combinations: Optional predefined qudit combinations.
    
    Returns:
        List of system dimension lists.
    """
    if qudit_combinations is not None:
        return [combo for combo in qudit_combinations if len(combo) == num_systems]
    
    combinations = []
    min_dim, max_dim = default_dim_range
    
    for dim in range(min_dim, max_dim + 1):
        combinations.append([dim] * num_systems)
    
    if num_systems >= 2:
        for dims in itertools.product(range(min_dim, max_dim + 1), repeat=num_systems):
            if len(set(dims)) > 1:
                combinations.append(list(dims))
    
    return combinations



_test_performance_results: Dict[str, Dict[str, Any]] = {}


def track_performance(test_name: Optional[str] = None):
    r"""Decorator to track performance of a test function.
    
    Usage::
    
        @track_performance("test_name")
        def test_xxx():
            ...
    """
    def decorator(func: Callable) -> Callable:
        name = test_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceMonitor() as monitor:
                result = func(*args, **kwargs)
            
            perf_data = monitor.to_dict()
            try:
                import inspect
                file_path = inspect.getfile(func)
                if 'tests' in file_path:
                    rel_path = file_path.split('tests' + os.sep)[-1] if 'tests' + os.sep in file_path else os.path.basename(file_path)
                    perf_data['file_name'] = rel_path
                else:
                    perf_data['file_name'] = os.path.basename(file_path)
            except Exception:
                perf_data['file_name'] = 'unknown'
            
            _test_performance_results[name] = perf_data
            return result
        
        return wrapper
    return decorator


def get_test_performance_results() -> Dict[str, Dict[str, Any]]:
    r"""Get all collected test performance results."""
    return _test_performance_results.copy()


def clear_test_performance_results():
    r"""Clear all collected test performance results."""
    global _test_performance_results
    _test_performance_results = {}
