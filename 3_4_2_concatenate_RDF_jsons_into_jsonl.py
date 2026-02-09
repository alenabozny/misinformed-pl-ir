#!/usr/bin/env python3
"""
Concatenate files that each contain a single JSON value into a single JSONL file.

Usage:
  python concat_json_to_jsonl.py [source_dir] [output.jsonl]

Defaults:
  source_dir = .
  output.jsonl = combined.jsonl
"""
import sys
from pathlib import Path
import json

def main(src_dir: Path, out_file: Path):
    src_dir = src_dir.resolve()
    out_file = out_file.resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_file.open("w", encoding="utf-8") as out:
        for p in sorted(src_dir.iterdir()):
            if not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8")
                obj = json.loads(text)
            except Exception as e:
                print(f"Skipping (invalid JSON): {p} — {e}", file=sys.stderr)
                continue
            # Write compact JSON on a single line
            out.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
            out.write("\n")
            count += 1

    print(f"Wrote {count} JSON lines to {out_file}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./triples/custom_concat.jsonl")
    main(src, out)