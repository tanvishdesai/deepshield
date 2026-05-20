from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from typing import TypeVar


T = TypeVar("T")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def log_config(title: str, values: Mapping[str, object]) -> None:
    log(title)
    for key, value in values.items():
        print(f"  - {key}: {value}", flush=True)


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "item",
    leave: bool = True,
    every: int | None = None,
) -> Iterator[T]:
    try:
        from tqdm.auto import tqdm

        yield from tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=leave)
        return
    except Exception:
        pass

    yield from _fallback_progress(iterable, total=total, desc=desc, unit=unit, every=every)


def _fallback_progress(
    iterable: Iterable[T],
    *,
    total: int | None,
    desc: str,
    unit: str,
    every: int | None,
) -> Iterator[T]:
    if total == 0:
        log(f"{desc}: nothing to do.")
        return

    count = 0
    interval = every or (max(total // 10, 1) if total else 1)
    if desc:
        if total:
            log(f"{desc}: starting ({total} {unit}{'' if total == 1 else 's'}).")
        else:
            log(f"{desc}: starting.")

    for item in iterable:
        count += 1
        if count == 1 or count % interval == 0 or (total is not None and count == total):
            if total is not None:
                log(f"{desc}: {count}/{total} {unit}{'' if count == 1 else 's'} complete.")
            else:
                log(f"{desc}: processed {count} {unit}{'' if count == 1 else 's'}.")
        yield item

    if desc:
        if total is not None:
            log(f"{desc}: completed ({count}/{total}).")
        else:
            log(f"{desc}: completed ({count} {unit}{'' if count == 1 else 's'}).")
