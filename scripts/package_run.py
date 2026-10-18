"""Build an allowlisted source bundle, optionally including exactly one checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import io

ROOT = Path(__file__).resolve().parents[1]


def package(output, wheel=None, checkpoint=None):
    files = [
        ROOT / "BoardDesign.csv",
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        ROOT / "native/Cargo.toml",
        ROOT / "native/Cargo.lock",
    ]
    files.extend((ROOT / "native/src").glob("*.rs"))
    for folder in ("src", "scripts", "tests"):
        files.extend((ROOT / folder).glob("*.py"))
    files.extend((ROOT / "scripts").glob("*.sh"))
    files.append(ROOT / "web/app.py")
    files.extend((ROOT / "web/static").glob("*.js"))
    files.append(ROOT / "tests/browser.test.cjs")
    entries = [(p, str(p.relative_to(ROOT))) for p in files]
    if wheel:
        entries.append((Path(wheel), "wheels/" + Path(wheel).name))
    if checkpoint:
        entries.append((Path(checkpoint), "models/latest.pt"))
    manifest = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in entries
    }
    with tarfile.open(output, "w:gz") as archive:
        for path, name in entries:
            archive.add(path, arcname=name, recursive=False)
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": Path(output).stat().st_size,
                "files": len(entries),
                "checkpoints": int(checkpoint is not None),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--wheel")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    package(args.output, args.wheel, args.checkpoint)
