"""Reproducibility records: environment, data fingerprint, run parameters.

Every run script writes one manifest next to its metrics so a number can always
be traced back to the bytes it was computed from. Hashing the whole 700 MB
train file on every run would be wasteful, so the fingerprint is
``(size, mtime, sha1 of the first 4 MiB)`` — enough to detect a different
delivery, cheap enough to do every time.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

from movie_rec.config import Settings

OFFICIAL_FILES = (
    "train",
    "validation",
    "movies",
    "candidates_val",
    "candidates_test",
    "cutoffs_val",
    "cutoffs_test",
)


def _package_versions() -> dict:
    versions = {}
    for name in ("pandas", "numpy", "scipy", "pyarrow", "pyspark"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:  # pragma: no cover - optional at import time
            versions[name] = None
    return versions


def _sha1_prefix(path: Path, size: int = 4 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        digest.update(handle.read(size))
    return digest.hexdigest()[:16]


def _total_ram_gb() -> float | None:
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 1e9, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:  # pragma: no cover - best effort only
        return None


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MiB (Windows and Linux)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            func = getattr(kernel32, "K32GetProcessMemoryInfo", None)
            if func is None:
                func = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            func.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            func.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            handle = wintypes.HANDLE(handle) if not isinstance(handle, int) else handle
            if not func(handle, ctypes.byref(counters), counters.cb):
                return None
            return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024, 1)
        return None
    except Exception:  # pragma: no cover - best effort only
        return None


def env_snapshot() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "total_ram_gb": _total_ram_gb(),
        "packages": _package_versions(),
        "java_home": os.environ.get("JAVA_HOME", ""),
        "pid": os.getpid(),
    }


def data_snapshot(settings: Settings) -> dict:
    out = {}
    for name in OFFICIAL_FILES:
        path = getattr(settings.official, name)
        if path.is_file():
            stat = path.stat()
            out[name] = {
                "path": str(path.relative_to(settings.project_root)),
                "bytes": stat.st_size,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "sha1_head": _sha1_prefix(path),
            }
        else:
            out[name] = {"path": str(path), "missing": True}
    return out


def build_manifest(settings: Settings, name: str, params: dict, extra: dict | None = None) -> dict:
    protocol = settings.protocol
    manifest = {
        "run": name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "protocol": {
            "positive_threshold": protocol.positive_threshold,
            "negatives_per_user": protocol.negatives_per_user,
            "val_seed": protocol.val_seed,
            "test_seed": protocol.test_seed,
            "ndcg_k": protocol.ndcg_k,
        },
        "env": env_snapshot(),
        "data": data_snapshot(settings),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
