#!/usr/bin/env python3
"""Mirror recorded evidence and source downloads into a static documentation tree.

Only files inside this repository are copied. No training, package installation,
credentials, Git commands, or remote actions are involved.
"""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/"docs/assets"
PUBLISHED_RESULT_DIRS=("figures", "torch_quantum", "torch_classical", "quantum", "classical",
    "finite_shot", "verification", "learning_curve", "physics_examples", "layer_forward",
    "pretrained_inference", "streaming_demo", "resources")


def recorded_results():
    """Leave user-created reproduction directories outside the published record."""
    yield from (path for path in sorted((ROOT/"results").iterdir()) if path.is_file())
    for name in PUBLISHED_RESULT_DIRS:
        yield from (path for path in sorted((ROOT/"results"/name).rglob("*")) if path.is_file())


def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    for source in recorded_results():
        if source.suffix in {".json", ".csv", ".npz", ".pt", ".txt"}:
            copy(source,ASSETS/"results"/source.relative_to(ROOT/"results"))
    for folder in ["verification","data"]:
        for source in sorted((ROOT/folder).rglob("*")):
            if source.is_file() and source.suffix in {".json",".csv",".npz",".pt"}:
                copy(source,ASSETS/folder/source.relative_to(ROOT/folder))
    seen={}
    for source in recorded_results():
        if source.is_file() and source.suffix in {".svg",".png",".pdf"}:
            destination=ASSETS/"figures"/source.name
            if source.name in seen and source.read_bytes()!=seen[source.name].read_bytes():
                raise ValueError(f"Two different figure files have the same name: {source.name}")
            seen[source.name]=source
            copy(source,destination)
    for folder in ["qem","scripts","examples","verification"]:
        for source in sorted((ROOT/folder).rglob("*.py")):
            copy(source,ASSETS/"code"/source.relative_to(ROOT))
    files=[]
    for path in sorted(ASSETS.rglob("*")):
        if path.is_file() and path != ASSETS/"manifest.json":
            files.append(dict(path=path.relative_to(ASSETS).as_posix(),bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (ASSETS/"manifest.json").write_text(json.dumps({"format":"qem-site-assets-v1","files":files},indent=2)+"\n")
    print(json.dumps({"asset_files":len(files),"bytes":sum(f["bytes"] for f in files)},indent=2))


if __name__=="__main__":
    main()
