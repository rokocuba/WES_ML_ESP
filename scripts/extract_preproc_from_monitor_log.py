#!/usr/bin/env python3
"""Extract Preproc28 frames from ESP-IDF monitor logs into image files."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

BEGIN_RE = re.compile(
    r"Preproc28 BEGIN name=(?P<name>[^ ]+) w=(?P<w>\d+) h=(?P<h>\d+) format=hex"
)
ROW_RE = re.compile(r"Preproc28 ROW(?P<row>\d{2}) (?P<hex>[0-9a-fA-F]+)")
END_RE = re.compile(r"Preproc28 END name=(?P<name>[^ ]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse monitor logs and export Preproc28 frames as image files."
    )
    parser.add_argument(
        "--log", type=Path, required=True, help="Path to saved monitor log text file"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("preproc_frames"), help="Output directory"
    )
    parser.add_argument(
        "--format",
        choices=("png", "pgm"),
        default="png",
        help="Output image format (default: png)",
    )
    return parser.parse_args()


def safe_stem(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "frame"


def write_pgm(path: Path, width: int, height: int, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        header = f"P5\n{width} {height}\n255\n".encode("ascii")
        f.write(header)
        f.write(data)


def write_png(path: Path, width: int, height: int, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.frombytes("L", (width, height), data)
    image.save(path, format="PNG")


def finalize_frame(
    out_dir: Path,
    counters: dict[str, int],
    name: str,
    width: int,
    height: int,
    rows: list[str],
    output_format: str,
) -> Path:
    if len(rows) != height:
        raise ValueError(f"Frame {name}: expected {height} rows, got {len(rows)}")

    for idx, row_hex in enumerate(rows):
        if len(row_hex) != width * 2:
            raise ValueError(
                f"Frame {name}: row {idx:02d} has {len(row_hex)} hex chars, expected {width * 2}"
            )

    raw = bytes.fromhex("".join(rows))
    key = safe_stem(name)
    counters[key] += 1
    out_path = out_dir / f"{key}_{counters[key]:03d}.{output_format}"
    if output_format == "png":
        write_png(out_path, width, height, raw)
    else:
        write_pgm(out_path, width, height, raw)
    return out_path


def main() -> None:
    args = parse_args()

    if not args.log.exists():
        raise SystemExit(f"Log file not found: {args.log}")

    counters: dict[str, int] = defaultdict(int)
    current_name: str | None = None
    current_width = 0
    current_height = 0
    current_rows: list[str] = []
    generated: list[Path] = []

    with args.log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            begin_match = BEGIN_RE.search(line)
            if begin_match:
                current_name = begin_match.group("name").strip()
                current_width = int(begin_match.group("w"))
                current_height = int(begin_match.group("h"))
                current_rows = []
                continue

            row_match = ROW_RE.search(line)
            if row_match and current_name is not None:
                current_rows.append(row_match.group("hex").strip().lower())
                continue

            end_match = END_RE.search(line)
            if end_match and current_name is not None:
                end_name = end_match.group("name").strip()
                if end_name != current_name:
                    raise ValueError(
                        f"Mismatched frame end: {end_name} != {current_name}"
                    )
                out_path = finalize_frame(
                    args.out_dir,
                    counters,
                    current_name,
                    current_width,
                    current_height,
                    current_rows,
                    args.format,
                )
                generated.append(out_path)
                current_name = None
                current_rows = []

    if current_name is not None:
        raise ValueError(f"Unclosed frame at end of log: {current_name}")

    if not generated:
        raise SystemExit("No Preproc28 frames found in the provided log")

    print(f"Extracted {len(generated)} frame(s) to {args.out_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
