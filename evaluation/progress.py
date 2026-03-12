from __future__ import annotations
from typing import Any, Iterable, Optional


def tqdm(iterable: Optional[Iterable[Any]] = None, **kwargs: Any):
    """
    tqdm wrapper with safe fallback.
    - tqdm(iterable, desc=..., total=...)
    - tqdm(total=..., desc=...)  # manual progress bar
    """
    try:
        from tqdm import tqdm as _tqdm  # type: ignore

        if iterable is None:
            return _tqdm(**kwargs)
        return _tqdm(iterable, **kwargs)
    except Exception:
        if iterable is None:

            class _Dummy:
                def update(self, n: int = 1) -> None: ...
                def close(self) -> None: ...
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return _Dummy()
        return iterable
