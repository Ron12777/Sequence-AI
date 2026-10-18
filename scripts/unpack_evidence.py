"""Restore losslessly compressed campaign evidence for local analysis."""

import gzip
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    restored = 0
    for folder in (root / "experiments", root / "artifacts/retained-campaign-final"):
        for source in sorted(folder.rglob("*.gz")):
            target = source.with_suffix("")
            if target.suffix not in {".json", ".jsonl", ".txt"}:
                continue
            content = gzip.decompress(source.read_bytes())
            if target.exists():
                if target.read_bytes() != content:
                    raise RuntimeError(f"Existing evidence differs: {target}")
                continue
            # Exclusive creation preserves any concurrently written evidence.
            with target.open("xb") as handle:
                handle.write(content)
            restored += 1
    print(f"Restored {restored} evidence files; existing files verified.")


if __name__ == "__main__":
    main()
