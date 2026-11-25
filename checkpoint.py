import os
from pathlib import Path
import shutil


def tensor_storage_bytes(value):
    import torch

    seen = set()

    def visit(item):
        if torch.is_tensor(item):
            storage = item.untyped_storage() if hasattr(item, "untyped_storage") else item.storage()
            key = (str(item.device), storage._cdata)
            if key in seen:
                return 0
            seen.add(key)
            return (
                storage.nbytes()
                if hasattr(storage, "nbytes")
                else storage.size() * item.element_size()
            )
        if isinstance(item, dict):
            return sum((visit(v) for v in item.values()))
        if isinstance(item, (list, tuple)):
            return sum((visit(v) for v in item))
        return 0

    return visit(value)


def atomic_save(state, path, save_fn=None):
    import torch

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Stale temporary checkpoint exists: {temporary}; inspect/remove it after confirming no other training process is active"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    estimate = max(tensor_storage_bytes(state), path.stat().st_size if path.exists() else 0)
    required = int(estimate * 1.05) + 64 * 1024**2
    free = shutil.disk_usage(path.parent).free
    if free < required:
        raise OSError(
            f"Insufficient free disk space for atomic checkpoint: need approximately {required / 1024 ** 3:.2f} GiB FREE, have {free / 1024 ** 3:.2f} GiB. Previous checkpoint preserved: {path}. Free space or increase storage quota. Only best_val.pt and best_test.pt are retained; a temporary full checkpoint still needs additional space during saving."
        )
    owned = False
    try:
        with temporary.open("xb") as stream:
            owned = True
            (save_fn or torch.save)(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException as exc:
        if owned and temporary.exists():
            temporary.unlink()
        if not isinstance(exc, Exception):
            raise
        free = shutil.disk_usage(path.parent).free
        raise RuntimeError(
            f"Checkpoint save failed; previous {path.name} preserved, partial temporary file removed. Free space now {free / 1024 ** 3:.2f} GiB. Check disk space, quota, and filesystem health. Original error: {exc}"
        ) from exc
