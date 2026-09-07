"""Hardware detection, PyTorch CUDA acceleration, and vectorized numerical backends for Phase 4 shifts."""

import os
import platform
from typing import Any, Dict, Optional, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False


def detect_hardware() -> Dict[str, Any]:
    """Detect platform, CPU, RAM, and PyTorch CUDA availability and properties."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
        "physical_cpus": psutil.cpu_count(logical=False) if PSUTIL_AVAILABLE else (os.cpu_count() or 1),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2) if PSUTIL_AVAILABLE else None,
        "torch_available": TORCH_AVAILABLE,
        "torch_version": torch.__version__ if TORCH_AVAILABLE else None,
        "cuda_available": bool(TORCH_AVAILABLE and torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if (TORCH_AVAILABLE and torch.cuda.is_available()) else None,
        "cuda_device_count": torch.cuda.device_count() if (TORCH_AVAILABLE and torch.cuda.is_available()) else 0,
        "gpu_model": torch.cuda.get_device_name(0) if (TORCH_AVAILABLE and torch.cuda.is_available()) else None,
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        if (TORCH_AVAILABLE and torch.cuda.is_available())
        else None,
    }
    return info


def should_use_gpu(
    n_rows: int,
    n_cols: int,
    backend: str = "auto",
    min_cells: int = 2000000,
) -> bool:
    """Decide whether to dispatch numerical transformation to GPU.
    
    Rules:
    - If backend == "numpy": always False.
    - If backend == "torch": True if CUDA is available, else False.
    - If backend == "auto": True if CUDA is available AND n_rows * n_cols >= min_cells.
    """
    cuda_ready = bool(TORCH_AVAILABLE and torch.cuda.is_available())
    if not cuda_ready:
        return False
    if backend == "numpy":
        return False
    if backend == "torch":
        return True
    if backend == "auto":
        return bool((n_rows * n_cols) >= min_cells)
    return False


def apply_affine_transform(
    X_arr: np.ndarray,
    scale: Union[float, np.ndarray],
    shift: Union[float, np.ndarray],
    use_gpu: bool = False,
    chunk_rows: int = 65536,
) -> np.ndarray:
    """Compute X * scale + shift elementwise in float64, with optional PyTorch CUDA acceleration.
    
    NaN values are preserved without corruption.
    """
    n_rows, n_cols = X_arr.shape
    X_mat = np.ascontiguousarray(X_arr, dtype=np.float64)
    if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            # Process in chunks if very large to prevent VRAM spikes
            if n_rows > chunk_rows:
                out = np.empty_like(X_mat, dtype=np.float64)
                for start in range(0, n_rows, chunk_rows):
                    end = min(start + chunk_rows, n_rows)
                    chunk = torch.from_numpy(X_mat[start:end].copy()).to(device=device, dtype=torch.float64)
                    s_tensor = torch.as_tensor(scale, device=device, dtype=torch.float64)
                    sh_tensor = torch.as_tensor(shift, device=device, dtype=torch.float64)
                    res = (chunk * s_tensor) + sh_tensor
                    out[start:end] = res.cpu().numpy()
                return out
            else:
                t_X = torch.from_numpy(X_mat.copy()).to(device=device, dtype=torch.float64)
                t_scale = torch.as_tensor(scale, device=device, dtype=torch.float64)
                t_shift = torch.as_tensor(shift, device=device, dtype=torch.float64)
                res = (t_X * t_scale) + t_shift
                return res.cpu().numpy()
        except Exception:
            # Fall back to NumPy on GPU failure (e.g. OOM)
            pass

    # NumPy CPU execution
    if n_rows > chunk_rows:
        out = np.empty_like(X_arr, dtype=np.float64)
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            out[start:end] = X_arr[start:end] * scale + shift
        return out
    return X_arr * scale + shift


def apply_additive_noise(
    X_arr: np.ndarray,
    noise_tensor: np.ndarray,
    scale: Union[float, np.ndarray],
    row_indices: Optional[np.ndarray] = None,
    use_gpu: bool = False,
    chunk_rows: int = 65536,
) -> np.ndarray:
    """Compute X + scale * noise_tensor, optionally applied only to selected row_indices."""
    n_rows, n_cols = X_arr.shape
    out = X_arr.copy().astype(np.float64)

    if row_indices is None:
        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                t_X = torch.from_numpy(out.copy()).to(device=device, dtype=torch.float64)
                t_noise = torch.from_numpy(noise_tensor.copy()).to(device=device, dtype=torch.float64)
                t_scale = torch.as_tensor(scale, device=device, dtype=torch.float64)
                res = t_X + t_scale * t_noise
                return res.cpu().numpy()
            except Exception:
                pass
        return out + scale * noise_tensor
    else:
        if len(row_indices) == 0:
            return out
        sub_X = out[row_indices]
        sub_noise = noise_tensor[row_indices] if noise_tensor.shape[0] == n_rows else noise_tensor
        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                t_sub = torch.from_numpy(sub_X.copy()).to(device=device, dtype=torch.float64)
                t_noise = torch.from_numpy(sub_noise.copy()).to(device=device, dtype=torch.float64)
                t_scale = torch.as_tensor(scale, device=device, dtype=torch.float64)
                res = t_sub + t_scale * t_noise
                out[row_indices] = res.cpu().numpy()
                return out
            except Exception:
                pass
        out[row_indices] = sub_X + scale * sub_noise
        return out


def apply_convex_interpolation(
    X_arr: np.ndarray,
    target_vectors: np.ndarray,
    alpha: float,
    row_indices: np.ndarray,
    use_gpu: bool = False,
) -> np.ndarray:
    """Compute (1 - alpha) * X + alpha * target_vectors for selected row_indices."""
    out = X_arr.copy().astype(np.float64)
    if len(row_indices) == 0:
        return out
    sub_X = out[row_indices]
    
    if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            t_sub = torch.from_numpy(sub_X.copy()).to(device=device, dtype=torch.float64)
            t_tgt = torch.from_numpy(target_vectors.copy()).to(device=device, dtype=torch.float64)
            res = (1.0 - alpha) * t_sub + alpha * t_tgt
            out[row_indices] = res.cpu().numpy()
            return out
        except Exception:
            pass
    out[row_indices] = (1.0 - alpha) * sub_X + alpha * target_vectors
    return out
