"""JSONC loading helpers shared by neural-network configuration modules."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _strip_jsonc_comments(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        index = 0
        while index < len(line):
            character = line[index]
            if escaped:
                escaped = False
            elif character == "\\" and in_string:
                escaped = True
            elif character == '"':
                in_string = not in_string
            elif character == "/" and not in_string and line[index:index + 2] == "//":
                line = line[:index]
                break
            index += 1
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def config_path_from_argv(
    argv: Optional[List[str]],
) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSONC configuration file downloaded from the RISE repository.",
    )
    arguments = sys.argv[1:] if argv is None else argv
    namespace, _ = parser.parse_known_args(arguments)
    return namespace.config.expanduser().resolve()


def load_jsonc_parameters(
    config_path: Path,
    path_keys: Iterable[str],
    allowed_keys: Iterable[str],
) -> Tuple[Dict[str, Any], Path]:
    path = config_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    values = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
    unknown = sorted(set(values) - set(allowed_keys))
    if unknown:
        raise ValueError(
            "Unknown neural-network configuration keys in {}: {}".format(
                path, ", ".join(unknown)
            )
        )
    for key in path_keys:
        value = values.get(key)
        if value is None:
            continue
        item = Path(value).expanduser()
        if not item.is_absolute():
            item = path.parent / item
        values[key] = item.resolve()
    return values, path
