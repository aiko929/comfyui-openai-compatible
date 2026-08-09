"""Remembers the last answer each node instance produced, so it can be reused on demand.

Kept on disk (not just in memory) so 'reuse_last_result' still works after ComfyUI restarts.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import folder_paths

_LOCK = threading.Lock()
_MAX_ENTRIES = 200  # plenty for normal use, and keeps the file from growing forever


def _path() -> str:
    directory = os.path.join(folder_paths.get_user_directory(), "openai_compatible")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "last_results.json")


def _load() -> dict:
    try:
        with open(_path(), encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        logging.warning("[openai-compatible] could not read stored results: %s", error)
        return {}
    return data if isinstance(data, dict) else {}


def make_key(workflow_id, node_id) -> str:
    """Scope by workflow so the same node id in two workflows keeps separate answers."""
    return f"{workflow_id or 'no-workflow'}:{node_id or 'no-node'}"


def get(key: str) -> str | None:
    with _LOCK:
        value = _load().get(key)
    return value if isinstance(value, str) else None


def put(key: str, text: str) -> None:
    with _LOCK:
        data = _load()
        data.pop(key, None)  # re-insert so the oldest entry is always first
        data[key] = text
        while len(data) > _MAX_ENTRIES:
            data.pop(next(iter(data)))

        path = _path()
        temp = f"{path}.tmp"
        try:
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=1)
            os.replace(temp, path)
        except OSError as error:
            logging.warning("[openai-compatible] could not store result: %s", error)
            try:
                os.remove(temp)
            except OSError:
                pass
